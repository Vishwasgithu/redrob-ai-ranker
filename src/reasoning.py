"""Grounded, deterministic reasoning for ranked candidates."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

_DELIVERY_PATTERN = re.compile(
    r"\b(?:built|shipped|deployed|production|serving|operated)\b", re.I
)
_RETRIEVAL_PATTERN = re.compile(
    r"\b(?:ranking|retrieval|search|recommendation|relevance|embedding|vector)\b",
    re.I,
)
_EVALUATION_PATTERN = re.compile(
    r"\b(?:evaluation|eval|ndcg|mrr|map|a/b test|offline|online metric)\b", re.I
)


def _matched_skills(row: pd.Series) -> list[str]:
    value: Any = row.get("matched_skills", ())
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return []


def generate_reasoning(row: pd.Series) -> str:
    """Generate at most two sentences using facts present in one candidate row."""
    title = str(row.get("current_title", "candidate")).strip() or "Candidate"
    company = str(row.get("current_company", "")).strip()
    years = float(row.get("years_of_experience", 0.0))
    skills = _matched_skills(row)[:3]
    text = str(row.get("candidate_text", ""))
    rank = int(row.get("rank", 100))

    if skills:
        skill_text = ", ".join(skills)
        first_templates = (
            f"{title} at {company} with {years:.1f} years of experience; relevant listed skills include {skill_text}.",
            f"With {years:.1f} years of experience, this {title} lists {skill_text}, directly relevant to the role.",
            f"The profile combines {years:.1f} years as a {title} with listed strengths in {skill_text}.",
        )
    else:
        first_templates = (
            f"{title} at {company} with {years:.1f} years of experience shows contextual alignment with the role.",
            f"This {title} brings {years:.1f} years of experience, although the explicit skill overlap is limited.",
            f"The profile has {years:.1f} years of experience as a {title} with more adjacent than exact skill evidence.",
        )
    first = first_templates[(rank - 1) % len(first_templates)]

    evidence: list[str] = []
    if _DELIVERY_PATTERN.search(text) and _RETRIEVAL_PATTERN.search(text):
        evidence.append("the career history describes production retrieval or ranking delivery")
    if _EVALUATION_PATTERN.search(text):
        evidence.append("it also references ranking evaluation or online testing")
    response = float(row.get("recruiter_response_rate", 0.0))
    github = float(row.get("github_activity_score", -1.0))
    notice = int(row.get("notice_period_days", 0))
    open_to_work = bool(row.get("open_to_work_flag", False))

    if evidence:
        second = evidence[0].capitalize()
        if len(evidence) > 1:
            second += ", and " + evidence[1]
        second += f"; recruiter response rate is {response:.0%}"
        if github >= 0:
            second += f" and GitHub activity is {github:.0f}/100"
        second += "."
    elif not open_to_work or notice > 90:
        concerns = []
        if not open_to_work:
            concerns.append("the profile is not marked open to work")
        if notice > 90:
            concerns.append(f"the stated notice period is {notice} days")
        second = "A practical concern is that " + " and ".join(concerns) + "."
    else:
        second = (
            f"The profile is open to work with a {notice}-day notice period and "
            f"a {response:.0%} recruiter response rate."
        )
    return f"{first} {second}"


def add_reasoning(ranked: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with grounded reasoning populated for every ranked row."""
    result = ranked.copy()
    result["reasoning"] = [
        generate_reasoning(row) for _, row in result.iterrows()
    ]
    return result
