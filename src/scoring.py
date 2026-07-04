"""Vectorized semantic, skill, experience, and behavioral scoring."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

SEMANTIC_WEIGHT = 0.45
SKILL_WEIGHT = 0.20
EXPERIENCE_WEIGHT = 0.15
BEHAVIOR_WEIGHT = 0.20

_ROLE_PATTERN = re.compile(
    r"\b(?:artificial intelligence|ai|machine learning|ml|nlp|search|"
    r"retrieval|recommendation|ranking|data scientist|applied scientist)\b",
    re.IGNORECASE,
)
_SENIOR_PATTERN = re.compile(
    r"\b(?:senior|staff|lead|principal|founding)\b", re.IGNORECASE
)
_PRODUCTION_PATTERN = re.compile(
    r"\b(?:built|shipped|deployed|production|serving|operated|scaled|"
    r"latency|a/b test|monitoring|pipeline|users|queries|index)\b",
    re.IGNORECASE,
)
_RANKING_PATTERN = re.compile(
    r"\b(?:ranking|retrieval|search|recommendation|relevance|bm25|vector|"
    r"embedding|ndcg|mrr|learning.to.rank|hybrid)\b",
    re.IGNORECASE,
)
_SERVICE_COMPANIES = {
    "accenture", "capgemini", "cognizant", "hcl", "infosys", "mindtree",
    "mphasis", "tcs", "tech mahindra", "wipro",
}
_COMPANY_FOUNDED = {
    "aganitha": 2017, "byju's": 2011, "cred": 2018, "dream11": 2008,
    "freshworks": 2010, "glance": 2019, "haptik": 2013, "inmobi": 2007,
    "krutrim": 2023, "meesho": 2015, "niramai": 2016, "nykaa": 2012,
    "ola": 2010, "paytm": 2010, "pharmeasy": 2015, "phonepe": 2015,
    "policybazaar": 2008, "rephrase.ai": 2019, "sarvam ai": 2023,
    "swiggy": 2014, "unacademy": 2015, "upgrad": 2015, "vedantu": 2011,
    "verloop.io": 2015, "wysa": 2015, "yellow.ai": 2016, "zomato": 2008,
}


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """Component scores plus evidence needed for explanations."""

    semantic: np.ndarray
    skill: np.ndarray
    experience: np.ndarray
    behavior: np.ndarray
    final: np.ndarray
    quality_factor: np.ndarray
    matched_skills: tuple[tuple[str, ...], ...]


def normalize_scores(
    values: Sequence[float] | np.ndarray,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    """Robustly map finite scores to [0, 1] without rank-order changes."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    finite = np.isfinite(array)
    result = np.zeros(array.shape, dtype=np.float32)
    if not finite.any():
        return result
    valid = array[finite]
    low, high = np.percentile(valid, [lower_percentile, upper_percentile])
    if math.isclose(float(low), float(high)):
        result[finite] = 0.5
        return result
    result[finite] = np.clip((valid - low) / (high - low), 0.0, 1.0)
    return result


def semantic_score(
    candidate_embeddings: np.ndarray,
    job_embedding: np.ndarray,
    requirement_embeddings: np.ndarray | None = None,
) -> np.ndarray:
    """Score normalized candidate vectors against the JD and requirements."""
    candidates = np.asarray(candidate_embeddings, dtype=np.float32)
    job = np.asarray(job_embedding, dtype=np.float32).reshape(-1)
    if candidates.ndim != 2 or candidates.shape[1] != job.shape[0]:
        raise ValueError("candidate and job embedding dimensions do not match")
    job_similarity = candidates @ job
    if requirement_embeddings is None or len(requirement_embeddings) == 0:
        raw = job_similarity
    else:
        requirements = np.asarray(requirement_embeddings, dtype=np.float32)
        if requirements.ndim != 2 or requirements.shape[1] != candidates.shape[1]:
            raise ValueError("requirement embedding dimensions do not match")
        similarities = candidates @ requirements.T
        top_count = min(3, similarities.shape[1])
        top = np.partition(similarities, -top_count, axis=1)[:, -top_count:]
        requirement_similarity = top.mean(axis=1)
        raw = 0.65 * job_similarity + 0.35 * requirement_similarity
    return normalize_scores(raw)


def _skill_key(value: str) -> str:
    return re.sub(r"[^a-z0-9+#]+", " ", value.casefold()).strip()


def skill_score(
    candidate_skills: Sequence[Sequence[str]],
    job_skills: Sequence[str],
    skill_vocabulary: Sequence[str],
    skill_embeddings: np.ndarray,
    job_skill_embeddings: np.ndarray,
) -> tuple[np.ndarray, tuple[tuple[str, ...], ...]]:
    """Combine exact skill coverage with semantic skill relatedness."""
    if not job_skills:
        return (
            np.full(len(candidate_skills), 0.5, dtype=np.float32),
            tuple(() for _ in candidate_skills),
        )
    vocabulary = list(skill_vocabulary)
    vectors = np.asarray(skill_embeddings, dtype=np.float32)
    targets = np.asarray(job_skill_embeddings, dtype=np.float32)
    if vectors.shape != (len(vocabulary), targets.shape[1]):
        raise ValueError("skill vocabulary and embedding shapes do not match")
    similarities = vectors @ targets.T
    best_similarity = similarities.max(axis=1)
    job_keys = {_skill_key(skill) for skill in job_skills}
    relevance: dict[str, float] = {}
    display_name: dict[str, str] = {}
    exact: dict[str, bool] = {}
    for index, skill in enumerate(vocabulary):
        key = _skill_key(skill)
        semantic = float(np.clip((best_similarity[index] - 0.20) / 0.65, 0, 1))
        is_exact = key in job_keys
        relevance[key] = 1.0 if is_exact else semantic
        display_name[key] = skill
        exact[key] = is_exact

    scores = np.zeros(len(candidate_skills), dtype=np.float32)
    all_matches: list[tuple[str, ...]] = []
    for row_index, skills in enumerate(candidate_skills):
        keys = [_skill_key(str(skill)) for skill in skills]
        ranked = sorted(
            ((relevance.get(key, 0.0), key) for key in keys), reverse=True
        )
        positive = [(value, key) for value, key in ranked if value >= 0.45]
        top = [value for value, _ in ranked[:5]]
        top_mean = float(np.mean(top)) if top else 0.0
        best = top[0] if top else 0.0
        exact_count = sum(exact.get(key, False) for key in keys)
        exact_coverage = min(exact_count / min(6, len(job_keys)), 1.0)
        scores[row_index] = np.clip(
            0.55 * top_mean + 0.25 * best + 0.20 * exact_coverage, 0, 1
        )
        all_matches.append(
            tuple(display_name[key] for _, key in positive[:5] if key in display_name)
        )
    return scores, tuple(all_matches)


def _safe_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _experience_band(years: np.ndarray) -> np.ndarray:
    below = np.clip(years / 5.0, 0.0, 1.0)
    ideal = np.ones_like(years)
    above = np.clip(1.0 - (years - 9.0) / 12.0, 0.35, 1.0)
    return np.where(years < 5.0, below, np.where(years <= 9.0, ideal, above))


def _record_quality(row: pd.Series) -> float:
    quality = 1.0
    career = _safe_json_list(row.get("career_history_json"))
    skills = _safe_json_list(row.get("skill_details_json"))
    claimed_months = float(row.get("years_of_experience", 0)) * 12.0
    career_months = sum(
        max(0, int(item.get("duration_months", 0)))
        for item in career if isinstance(item, dict)
    )
    if claimed_months > 0 and abs(career_months - claimed_months) > max(
        24.0, claimed_months * 0.45
    ):
        quality *= 0.55

    current_count = sum(
        bool(item.get("is_current")) for item in career if isinstance(item, dict)
    )
    if current_count != 1:
        quality *= 0.7

    impossible_jobs = 0
    for item in career:
        if not isinstance(item, dict):
            continue
        company = str(item.get("company", "")).casefold()
        founded = _COMPANY_FOUNDED.get(company)
        start_text = str(item.get("start_date", ""))
        if founded and len(start_text) >= 4 and start_text[:4].isdigit():
            if int(start_text[:4]) < founded:
                impossible_jobs += 1
        duration = int(item.get("duration_months", 0) or 0)
        start_year = int(start_text[:4]) if start_text[:4].isdigit() else None
        if start_year and duration > (2027 - start_year) * 12 + 3:
            impossible_jobs += 1
    if impossible_jobs:
        quality *= max(0.2, 1.0 - 0.4 * impossible_jobs)

    expert = [
        item for item in skills
        if isinstance(item, dict) and item.get("proficiency") == "expert"
    ]
    contradictory = sum(
        item.get("duration_months") is not None
        and int(item.get("duration_months") or 0) < 6
        for item in expert
    )
    if len(expert) >= 8 and contradictory / len(expert) >= 0.5:
        quality *= 0.25
    elif contradictory >= 3:
        quality *= 0.65
    return float(np.clip(quality, 0.1, 1.0))


def experience_score(dataframe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Score experience range, role relevance, shipping evidence, and integrity."""
    years = dataframe["years_of_experience"].to_numpy(dtype=np.float64)
    band = _experience_band(years)
    texts = dataframe["candidate_text"].fillna("").astype(str)
    titles = dataframe["current_title"].fillna("").astype(str)
    role = np.array(
        [
            1.0 if _ROLE_PATTERN.search(title) and _SENIOR_PATTERN.search(title)
            else 0.88 if _ROLE_PATTERN.search(title)
            else 0.35 if re.search(r"software|data engineer", title, re.I)
            else 0.0
            for title in titles
        ],
        dtype=np.float64,
    )
    production = np.array(
        [min(len(_PRODUCTION_PATTERN.findall(text)) / 8.0, 1.0) for text in texts],
        dtype=np.float64,
    )
    ranking_depth = np.array(
        [min(len(_RANKING_PATTERN.findall(text)) / 8.0, 1.0) for text in texts],
        dtype=np.float64,
    )
    product = np.ones(len(dataframe), dtype=np.float64)
    for index, companies_value in enumerate(dataframe["career_companies_json"]):
        companies = [str(value).casefold() for value in _safe_json_list(companies_value)]
        if companies and all(company in _SERVICE_COMPANIES for company in companies):
            product[index] = 0.1
    raw = 0.40 * band + 0.25 * role + 0.20 * production + 0.10 * ranking_depth + 0.05 * product
    quality = np.array(
        [_record_quality(row) for _, row in dataframe.iterrows()], dtype=np.float32
    )
    return np.clip(raw, 0, 1).astype(np.float32), quality


def _percentile_feature(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0)
    transformed = np.log1p(numeric.to_numpy(dtype=np.float64))
    return normalize_scores(transformed, lower_percentile=0, upper_percentile=95)


def behavior_score(dataframe: pd.DataFrame) -> np.ndarray:
    """Combine availability and engagement without using salary as a proxy."""
    github = pd.to_numeric(dataframe["github_activity_score"], errors="coerce").to_numpy()
    github = np.where(github < 0, 35.0, github) / 100.0
    response = dataframe["recruiter_response_rate"].to_numpy(dtype=np.float64)
    completeness = dataframe["profile_completeness_score"].to_numpy(dtype=np.float64) / 100.0
    saved = _percentile_feature(dataframe["saved_by_recruiters_30d"])
    appearance = _percentile_feature(dataframe["search_appearance_30d"])
    interview = dataframe["interview_completion_rate"].to_numpy(dtype=np.float64)
    offer = dataframe["offer_acceptance_rate"].to_numpy(dtype=np.float64)
    offer = np.where(offer < 0, 0.5, offer)
    open_to_work = dataframe["open_to_work_flag"].astype(float).to_numpy()
    notice = dataframe["notice_period_days"].to_numpy(dtype=np.float64)
    notice = np.clip(1.0 - np.maximum(notice - 15.0, 0.0) / 165.0, 0.0, 1.0)
    relocate = dataframe["willing_to_relocate"].astype(float).to_numpy()

    score = (
        0.14 * github + 0.18 * response + 0.09 * completeness
        + 0.09 * saved + 0.06 * appearance + 0.12 * interview
        + 0.07 * offer + 0.12 * open_to_work + 0.08 * notice
        + 0.05 * relocate
    )
    return np.clip(score, 0, 1).astype(np.float32)


def hybrid_score(
    semantic: Sequence[float],
    skill: Sequence[float],
    experience: Sequence[float],
    behavior: Sequence[float],
) -> np.ndarray:
    """Apply the challenge-defined weighted hybrid scoring formula."""
    arrays = [np.asarray(value, dtype=np.float32) for value in (
        semantic, skill, experience, behavior
    )]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("all score components must have the same shape")
    result = (
        SEMANTIC_WEIGHT * arrays[0] + SKILL_WEIGHT * arrays[1]
        + EXPERIENCE_WEIGHT * arrays[2] + BEHAVIOR_WEIGHT * arrays[3]
    )
    return np.clip(result, 0, 1).astype(np.float32)


def score_candidates(
    dataframe: pd.DataFrame,
    candidate_embeddings: np.ndarray,
    job_embedding: np.ndarray,
    requirement_embeddings: np.ndarray,
    job_skills: Sequence[str],
    skill_vocabulary: Sequence[str],
    skill_embeddings: np.ndarray,
    job_skill_embeddings: np.ndarray,
) -> ScoreResult:
    """Compute every score and apply record-integrity penalties consistently."""
    semantic = semantic_score(
        candidate_embeddings, job_embedding, requirement_embeddings
    )
    candidate_skills = [
        [str(skill) for skill in _safe_json_list(value)]
        for value in dataframe["skill_names_json"]
    ]
    skills, matched = skill_score(
        candidate_skills,
        job_skills,
        skill_vocabulary,
        skill_embeddings,
        job_skill_embeddings,
    )
    experience, quality = experience_score(dataframe)
    behavior = behavior_score(dataframe)

    semantic = semantic * quality
    skills = skills * quality
    experience = experience * quality
    behavior = behavior * quality
    final = hybrid_score(semantic, skills, experience, behavior)
    return ScoreResult(
        semantic=semantic,
        skill=skills,
        experience=experience,
        behavior=behavior,
        final=final,
        quality_factor=quality,
        matched_skills=matched,
    )


def build_explainability(result: ScoreResult) -> list[str]:
    """Serialize transparent component-level explanations for every row."""
    records: list[str] = []
    for index in range(len(result.final)):
        records.append(json.dumps({
            "Semantic Score": round(float(result.semantic[index]), 6),
            "Skill Score": round(float(result.skill[index]), 6),
            "Experience Score": round(float(result.experience[index]), 6),
            "Behavior Score": round(float(result.behavior[index]), 6),
            "Final Score": round(float(result.final[index]), 6),
            "Data Quality Factor": round(float(result.quality_factor[index]), 6),
            "Matched Skills": list(result.matched_skills[index]),
        }, ensure_ascii=False, separators=(",", ":")))
    return records
