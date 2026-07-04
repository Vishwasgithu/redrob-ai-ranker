"""Deterministic candidate ranking and submission export."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

REQUIRED_SUBMISSION_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]


def rank_candidates(
    candidates: pd.DataFrame,
    *,
    score_column: str = "final_score",
    top_k: int = 100,
) -> pd.DataFrame:
    """Stable-sort all candidates and return unique sequential top-k ranks."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    required = {"candidate_id", score_column}
    missing = required.difference(candidates.columns)
    if missing:
        raise KeyError(f"missing ranking columns: {sorted(missing)}")
    if len(candidates) < top_k:
        raise ValueError(
            f"cannot return top {top_k} from only {len(candidates)} candidates"
        )
    if candidates["candidate_id"].duplicated().any():
        raise ValueError("candidate_id values must be unique")
    scores = pd.to_numeric(candidates[score_column], errors="coerce")
    if scores.isna().any():
        raise ValueError(f"{score_column} contains missing or non-numeric values")

    ranked = candidates.assign(**{score_column: scores}).sort_values(
        by=[score_column, "candidate_id"],
        ascending=[False, True],
        kind="mergesort",
        ignore_index=True,
    )
    ranked.insert(1, "rank", range(1, len(ranked) + 1))
    return ranked.iloc[:top_k].copy()


def export_submission(ranked: pd.DataFrame, path: str | Path) -> Path:
    """Write an exact validator-compatible UTF-8 CSV atomically."""
    missing = set(REQUIRED_SUBMISSION_COLUMNS).difference(ranked.columns)
    if missing:
        raise KeyError(f"missing submission columns: {sorted(missing)}")
    submission = ranked.loc[:, REQUIRED_SUBMISSION_COLUMNS].copy()
    if len(submission) != 100:
        raise ValueError("submission must contain exactly 100 candidates")
    submission["rank"] = submission["rank"].astype(int)
    submission["score"] = pd.to_numeric(submission["score"]).map(
        lambda value: f"{value:.6f}"
    )

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".tmp.csv", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        submission.to_csv(
            temporary, index=False, encoding="utf-8", lineterminator="\n"
        )
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination
