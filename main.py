"""End-to-end Redrob semantic candidate-ranking pipeline."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import itertools
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.embed import (
    DEFAULT_MODEL_NAME,
    create_embedding_bundle_from_parquet,
    get_or_create_embeddings,
)
from src.load_data import load_job_description, load_schema, stream_candidates
from src.preprocess import (
    extract_candidate_dataframe,
    extract_job_requirements,
    extract_job_skills,
    preprocess_job_description,
)
from src.ranking import export_submission, rank_candidates
from src.reasoning import add_reasoning
from src.scoring import build_explainability, score_candidates
from src.utils import load_dataframe, save_dataframe, seed_everything, setup_logger

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_VERSION = 3


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Rank the top 100 Redrob candidates for a supplied JD."
    )
    parser.add_argument("--candidates", type=Path, default=PROJECT_ROOT / "data" / "candidates.jsonl")
    parser.add_argument("--schema", type=Path, default=PROJECT_ROOT / "data" / "candidate_schema.json")
    parser.add_argument("--job-description", type=Path, default=PROJECT_ROOT / "data" / "job_description.docx")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "submission.csv")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "outputs" / "cache")
    parser.add_argument("--model-cache", type=Path, default=PROJECT_ROOT / "models" / "huggingface")
    parser.add_argument("--ranked-output", type=Path, default=PROJECT_ROOT / "outputs" / "ranked_candidates.parquet")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--expected-candidates", type=int, default=100_000)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rebuild preprocessing and embedding caches.")
    parser.add_argument("--local-files-only", action="store_true", help="Never download model files.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    arguments = parser.parse_args(argv)
    if arguments.batch_size < 1:
        parser.error("--batch-size must be positive")
    if arguments.expected_candidates < 100:
        parser.error("--expected-candidates must be at least 100")
    return arguments


def _source_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _load_or_preprocess(
    arguments: argparse.Namespace,
    logger: Any,
) -> Path:
    cache_dir = arguments.cache_dir.expanduser().resolve()
    processed_path = cache_dir / "processed_candidates.parquet"
    manifest_path = cache_dir / "preprocessing_manifest.json"
    expected_manifest = {
        "cache_version": CACHE_VERSION,
        "candidates": _source_fingerprint(arguments.candidates),
        "schema": _source_fingerprint(arguments.schema),
    }
    required_columns = {
        "candidate_id", "candidate_text", "skill_names_json",
        "skill_details_json", "career_history_json", "career_companies_json",
    }
    if not arguments.force and processed_path.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest == expected_manifest:
                cached_columns = set(pq.ParquetFile(processed_path).schema.names)
                if required_columns.issubset(cached_columns):
                    logger.info("Using cached preprocessing from %s", processed_path)
                    return processed_path
        except (OSError, ValueError, json.JSONDecodeError):
            logger.warning("Ignoring invalid preprocessing cache")

    schema = load_schema(arguments.schema)
    candidates = stream_candidates(
        arguments.candidates,
        schema=schema,
        validate=not arguments.skip_validation,
        show_progress=True,
        total=arguments.expected_candidates,
    )
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".processed_candidates.", suffix=".tmp.parquet",
        dir=processed_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    temporary_path.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    seen_ids: set[str] = set()
    try:
        while True:
            batch = list(itertools.islice(candidates, 2_000))
            if not batch:
                break
            chunk = extract_candidate_dataframe(batch, show_progress=False)
            ids = set(chunk["candidate_id"].astype(str))
            if len(ids) != len(chunk) or seen_ids.intersection(ids):
                raise RuntimeError("candidate dataset contains duplicate IDs")
            seen_ids.update(ids)
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_path, table.schema, compression="zstd"
                )
            writer.write_table(table)
            total_rows += len(chunk)
        if writer is None or total_rows == 0:
            raise RuntimeError("no candidates were loaded")
        writer.close()
        writer = None
        os.replace(temporary_path, processed_path)
    except Exception:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(expected_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return processed_path


def _json_string_lists(values: pd.Series) -> list[list[str]]:
    result: list[list[str]] = []
    for value in values:
        parsed = json.loads(str(value))
        if not isinstance(parsed, list):
            raise ValueError("cached skill_names_json is not a list")
        result.append([str(item) for item in parsed])
    return result


def _validate_submission(path: Path, candidate_ids: set[str]) -> None:
    validator_path = PROJECT_ROOT / "data" / "validate_submission.py"
    spec = importlib.util.spec_from_file_location("redrob_validator", validator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to import validator from {validator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    errors = module.validate_submission(path)
    submission = pd.read_csv(path, encoding="utf-8")
    unknown = set(submission["candidate_id"]) - candidate_ids
    if unknown:
        errors.append(f"Unknown candidate IDs: {sorted(unknown)}")
    if errors:
        raise RuntimeError("submission validation failed:\n- " + "\n- ".join(errors))


def run_pipeline(arguments: argparse.Namespace) -> dict[str, float | int | str]:
    """Execute preprocessing, semantic ranking, export, and validation."""
    logger = setup_logger(level=arguments.log_level)
    seed_everything(arguments.seed)
    started_at = time.perf_counter()
    os.environ.setdefault("HF_HOME", str(arguments.model_cache.expanduser().resolve()))

    raw_job_description = load_job_description(arguments.job_description)
    normalized_job_description = preprocess_job_description(raw_job_description)
    processed_path = _load_or_preprocess(arguments, logger)
    skill_frame = pd.read_parquet(
        processed_path,
        columns=["skill_names_json"],
        engine="pyarrow",
        dtype_backend="pyarrow",
    )
    candidate_skill_lists = _json_string_lists(skill_frame["skill_names_json"])
    skill_vocabulary = sorted(
        {skill for skills in candidate_skill_lists for skill in skills},
        key=str.casefold,
    )
    del candidate_skill_lists
    job_skills = extract_job_skills(raw_job_description, skill_vocabulary)
    requirements = extract_job_requirements(raw_job_description)
    role_line = raw_job_description.splitlines()[0].strip()
    semantic_target = (
        f"{role_line}. Required capabilities: {', '.join(job_skills)}. "
        + " ".join(requirements[:8])
    )
    logger.info(
        "Scoring %d candidates against %d extracted requirements and %d JD skills",
        len(skill_frame), len(requirements), len(job_skills),
    )

    bundle = create_embedding_bundle_from_parquet(
        processed_path,
        semantic_target or normalized_job_description,
        requirements,
        cache_dir=arguments.cache_dir / "embeddings",
        model_name=arguments.model_name,
        batch_size=arguments.batch_size,
        device=arguments.device,
        force=arguments.force,
        local_files_only=arguments.local_files_only,
    )
    gc.collect()
    all_skill_texts = sorted(
        set(skill_vocabulary).union(job_skills), key=str.casefold
    )
    all_skill_embeddings = get_or_create_embeddings(
        all_skill_texts,
        arguments.cache_dir / "embeddings" / "skill_embeddings.npy",
        model_name=arguments.model_name,
        batch_size=arguments.batch_size,
        device=arguments.device,
        force=arguments.force,
        local_files_only=arguments.local_files_only,
    )
    del skill_frame
    gc.collect()
    embedding_by_skill = {
        skill: all_skill_embeddings[index]
        for index, skill in enumerate(all_skill_texts)
    }
    vocabulary_vectors = np.stack(
        [embedding_by_skill[skill] for skill in skill_vocabulary]
    )
    job_skill_vectors = (
        np.stack([embedding_by_skill[skill] for skill in job_skills])
        if job_skills else vocabulary_vectors[:1]
    )

    dataframe = load_dataframe(processed_path)

    result = score_candidates(
        dataframe,
        bundle.candidates,
        bundle.job_description,
        bundle.requirements,
        job_skills,
        skill_vocabulary,
        vocabulary_vectors,
        job_skill_vectors,
    )
    scored = dataframe.copy()
    scored["semantic_score"] = result.semantic
    scored["skill_score"] = result.skill
    scored["experience_score"] = result.experience
    scored["behavior_score"] = result.behavior
    scored["final_score"] = result.final
    scored["data_quality_factor"] = result.quality_factor
    scored["matched_skills"] = [list(value) for value in result.matched_skills]
    scored["matched_skills_json"] = [
        json.dumps(list(value), ensure_ascii=False) for value in result.matched_skills
    ]
    scored["explainability"] = build_explainability(result)

    ranked_all = rank_candidates(scored, top_k=len(scored))
    save_dataframe(ranked_all.drop(columns=["matched_skills"]), arguments.ranked_output)
    top = add_reasoning(ranked_all.iloc[:100].copy())
    top["score"] = top["final_score"]
    output_path = export_submission(top, arguments.output)
    _validate_submission(output_path, set(dataframe["candidate_id"].astype(str)))

    elapsed = time.perf_counter() - started_at
    statistics: dict[str, float | int | str] = {
        "total_candidates": len(dataframe),
        "processing_seconds": elapsed,
        "top_candidate": str(top.iloc[0]["candidate_id"]),
        "average_score": float(top["final_score"].mean()),
    }
    logger.info("Validated submission written to %s", output_path)
    print(f"Total candidates: {statistics['total_candidates']:,}")
    print(f"Processing time: {statistics['processing_seconds']:.2f} seconds")
    print(f"Top candidate: {statistics['top_candidate']}")
    print(f"Average score: {statistics['average_score']:.6f}")
    print("Submission validation: passed")
    return statistics


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process-compatible exit code."""
    arguments = parse_arguments(argv)
    logger = setup_logger(level=arguments.log_level)
    try:
        run_pipeline(arguments)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted")
        return 130
    except Exception:
        logger.exception("Candidate-ranking pipeline failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
