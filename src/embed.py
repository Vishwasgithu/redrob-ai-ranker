"""Sentence-Transformer encoding with deterministic, validated disk caches."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be generated or safely loaded."""


@dataclass(frozen=True, slots=True)
class EmbeddingBundle:
    """All semantic vectors needed by the hybrid ranker."""

    candidates: np.ndarray
    job_description: np.ndarray
    requirements: np.ndarray
    requirement_texts: tuple[str, ...]
    model_name: str


def resolve_device(requested: str = "auto") -> str:
    """Return a supported torch device, preferring CUDA when requested."""
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    try:
        import torch
    except ImportError as exc:
        raise EmbeddingError("PyTorch is required for sentence embeddings") from exc

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA was requested but is unavailable; using CPU")
        return "cpu"
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        LOGGER.warning("MPS was requested but is unavailable; using CPU")
        return "cpu"
    return requested


def load_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    device: str = "auto",
    local_files_only: bool = False,
) -> Any:
    """Load the configured SentenceTransformer lazily."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingError(
            "sentence-transformers is not installed; install requirements.txt"
        ) from exc

    resolved_device = resolve_device(device)
    local_snapshot = (
        Path(__file__).resolve().parents[1] / "models" / "all-mpnet-base-v2"
    )
    model_source = (
        str(local_snapshot)
        if model_name == DEFAULT_MODEL_NAME and local_snapshot.is_dir()
        else model_name
    )
    LOGGER.info("Loading embedding model %s on %s", model_source, resolved_device)
    try:
        return SentenceTransformer(
            model_source,
            device=resolved_device,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise EmbeddingError(
            f"Unable to load embedding model {model_name!r}: {exc}"
        ) from exc


def generate_embeddings(
    texts: Sequence[str],
    model: Any,
    *,
    batch_size: int = 128,
    show_progress: bool = True,
) -> np.ndarray:
    """Batch-encode text as normalized float32 embeddings."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not texts:
        dimension = int(model.get_sentence_embedding_dimension())
        return np.empty((0, dimension), dtype=np.float32)
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("all embedding inputs must be non-empty strings")

    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    result = np.asarray(vectors, dtype=np.float32)
    if result.ndim != 2 or result.shape[0] != len(texts):
        raise EmbeddingError(
            f"Model returned shape {result.shape} for {len(texts)} texts"
        )
    if not np.isfinite(result).all():
        raise EmbeddingError("Model returned non-finite embeddings")
    return result


def _text_fingerprint(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _atomic_save_array(array: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".tmp.npy", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.save(temporary, np.asarray(array, dtype=np.float32), allow_pickle=False)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_save_json(value: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".tmp.json", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def save_embeddings(
    embeddings: np.ndarray,
    path: str | Path,
    *,
    model_name: str,
    texts: Sequence[str],
) -> Path:
    """Atomically persist embeddings and integrity metadata."""
    destination = Path(path).expanduser().resolve()
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != len(texts):
        raise ValueError("embedding rows must match text count")
    _atomic_save_array(array, destination)
    _atomic_save_json(
        {
            "model_name": model_name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "text_fingerprint": _text_fingerprint(texts),
        },
        destination.with_suffix(destination.suffix + ".json"),
    )
    return destination


def load_embeddings(
    path: str | Path,
    *,
    model_name: str,
    texts: Sequence[str],
    mmap_mode: str | None = "r",
) -> np.ndarray | None:
    """Load a cache only when model, shape, and text fingerprint match."""
    source = Path(path).expanduser().resolve()
    metadata_path = source.with_suffix(source.suffix + ".json")
    if not source.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("model_name") != model_name:
            return None
        if metadata.get("text_fingerprint") != _text_fingerprint(texts):
            return None
        array = np.load(source, mmap_mode=mmap_mode, allow_pickle=False)
        expected_shape = tuple(metadata.get("shape", ()))
        if array.ndim != 2 or array.shape != expected_shape:
            return None
        if array.shape[0] != len(texts):
            return None
        return array
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        LOGGER.warning("Ignoring invalid embedding cache at %s", source)
        return None


def create_embedding_bundle(
    candidate_texts: Sequence[str],
    job_description: str,
    requirements: Sequence[str],
    *,
    cache_dir: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 128,
    device: str = "auto",
    force: bool = False,
    local_files_only: bool = False,
) -> EmbeddingBundle:
    """Load or generate all candidate, JD, and requirement embeddings."""
    cache = Path(cache_dir).expanduser().resolve()
    candidate_path = cache / "candidate_embeddings.npy"
    job_path = cache / "job_embedding.npy"
    requirement_path = cache / "requirement_embeddings.npy"
    job_texts = [job_description]
    requirement_texts = list(requirements) or [job_description]

    candidate_vectors = None if force else load_embeddings(
        candidate_path, model_name=model_name, texts=candidate_texts
    )
    job_vectors = None if force else load_embeddings(
        job_path, model_name=model_name, texts=job_texts
    )
    requirement_vectors = None if force else load_embeddings(
        requirement_path, model_name=model_name, texts=requirement_texts
    )

    if all(value is not None for value in (
        candidate_vectors, job_vectors, requirement_vectors
    )):
        LOGGER.info("Using validated embedding caches from %s", cache)
    else:
        model = load_embedding_model(
            model_name, device=device, local_files_only=local_files_only
        )
        if candidate_vectors is None:
            candidate_vectors = generate_embeddings(
                candidate_texts, model, batch_size=batch_size
            )
            save_embeddings(
                candidate_vectors,
                candidate_path,
                model_name=model_name,
                texts=candidate_texts,
            )
        if job_vectors is None:
            job_vectors = generate_embeddings(job_texts, model, show_progress=False)
            save_embeddings(
                job_vectors, job_path, model_name=model_name, texts=job_texts
            )
        if requirement_vectors is None:
            requirement_vectors = generate_embeddings(
                requirement_texts, model, batch_size=batch_size, show_progress=False
            )
            save_embeddings(
                requirement_vectors,
                requirement_path,
                model_name=model_name,
                texts=requirement_texts,
            )

    assert candidate_vectors is not None
    assert job_vectors is not None
    assert requirement_vectors is not None
    return EmbeddingBundle(
        candidates=np.asarray(candidate_vectors),
        job_description=np.asarray(job_vectors[0]),
        requirements=np.asarray(requirement_vectors),
        requirement_texts=tuple(requirement_texts),
        model_name=model_name,
    )


def _parquet_fingerprint(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    parquet = pq.ParquetFile(path)
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "rows": parquet.metadata.num_rows,
    }


def _load_streamed_cache(
    path: Path,
    *,
    model_name: str,
    source_fingerprint: dict[str, int | str],
) -> np.ndarray | None:
    metadata_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("model_name") != model_name:
            return None
        if metadata.get("source_fingerprint") != source_fingerprint:
            return None
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != metadata.get("shape"):
            return None
        return array
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def create_embedding_bundle_from_parquet(
    parquet_path: str | Path,
    job_description: str,
    requirements: Sequence[str],
    *,
    cache_dir: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 128,
    device: str = "auto",
    force: bool = False,
    local_files_only: bool = False,
) -> EmbeddingBundle:
    """Stream candidate text from Parquet so model and profiles never co-reside."""
    source = Path(parquet_path).expanduser().resolve()
    cache = Path(cache_dir).expanduser().resolve()
    candidate_path = cache / "candidate_embeddings.npy"
    job_path = cache / "job_embedding.npy"
    requirement_path = cache / "requirement_embeddings.npy"
    source_fingerprint = _parquet_fingerprint(source)
    job_texts = [job_description]
    requirement_texts = list(requirements) or [job_description]

    candidate_vectors = None if force else _load_streamed_cache(
        candidate_path,
        model_name=model_name,
        source_fingerprint=source_fingerprint,
    )
    job_vectors = None if force else load_embeddings(
        job_path, model_name=model_name, texts=job_texts
    )
    requirement_vectors = None if force else load_embeddings(
        requirement_path, model_name=model_name, texts=requirement_texts
    )
    if all(value is not None for value in (
        candidate_vectors, job_vectors, requirement_vectors
    )):
        LOGGER.info("Using validated embedding caches from %s", cache)
    else:
        model = load_embedding_model(
            model_name, device=device, local_files_only=local_files_only
        )
        if candidate_vectors is None:
            cache.mkdir(parents=True, exist_ok=True)
            rows = int(source_fingerprint["rows"])
            dimension = int(model.get_sentence_embedding_dimension())
            temporary = candidate_path.with_name(
                f".{candidate_path.stem}.{os.getpid()}.tmp.npy"
            )
            temporary.unlink(missing_ok=True)
            output = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=np.float32,
                shape=(rows, dimension),
            )
            parquet = pq.ParquetFile(source)
            offset = 0
            progress = tqdm(
                total=rows,
                desc="Embedding candidates",
                unit="candidate",
                dynamic_ncols=True,
            )
            try:
                for record_batch in parquet.iter_batches(
                    batch_size=batch_size,
                    columns=["candidate_text"],
                    use_threads=True,
                ):
                    texts = [str(value) for value in record_batch.column(0).to_pylist()]
                    encoded = generate_embeddings(
                        texts, model, batch_size=batch_size, show_progress=False
                    )
                    output[offset : offset + len(encoded)] = encoded
                    offset += len(encoded)
                    progress.update(len(encoded))
                if offset != rows:
                    raise EmbeddingError(
                        f"Encoded {offset} candidates but Parquet contains {rows}"
                    )
                output.flush()
                del output
                output = None
                os.replace(temporary, candidate_path)
                _atomic_save_json(
                    {
                        "model_name": model_name,
                        "shape": [rows, dimension],
                        "dtype": "float32",
                        "source_fingerprint": source_fingerprint,
                    },
                    candidate_path.with_suffix(candidate_path.suffix + ".json"),
                )
                candidate_vectors = np.load(
                    candidate_path, mmap_mode="r", allow_pickle=False
                )
            except Exception:
                if output is not None:
                    del output
                temporary.unlink(missing_ok=True)
                raise
            finally:
                progress.close()
        if job_vectors is None:
            job_vectors = generate_embeddings(job_texts, model, show_progress=False)
            save_embeddings(
                job_vectors, job_path, model_name=model_name, texts=job_texts
            )
        if requirement_vectors is None:
            requirement_vectors = generate_embeddings(
                requirement_texts, model, batch_size=batch_size, show_progress=False
            )
            save_embeddings(
                requirement_vectors,
                requirement_path,
                model_name=model_name,
                texts=requirement_texts,
            )

    assert candidate_vectors is not None
    assert job_vectors is not None
    assert requirement_vectors is not None
    return EmbeddingBundle(
        candidates=np.asarray(candidate_vectors),
        job_description=np.asarray(job_vectors[0]),
        requirements=np.asarray(requirement_vectors),
        requirement_texts=tuple(requirement_texts),
        model_name=model_name,
    )


def get_or_create_embeddings(
    texts: Sequence[str],
    path: str | Path,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 128,
    device: str = "auto",
    force: bool = False,
    local_files_only: bool = False,
) -> np.ndarray:
    """Load a validated arbitrary-text cache or generate it."""
    if not force:
        cached = load_embeddings(path, model_name=model_name, texts=texts)
        if cached is not None:
            return np.asarray(cached)
    model = load_embedding_model(
        model_name, device=device, local_files_only=local_files_only
    )
    embeddings = generate_embeddings(
        texts, model, batch_size=batch_size, show_progress=False
    )
    save_embeddings(embeddings, path, model_name=model_name, texts=texts)
    return embeddings
