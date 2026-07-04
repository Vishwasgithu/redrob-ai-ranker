"""Candidate and job-description preprocessing for semantic retrieval."""

from __future__ import annotations

import html
import json
import logging
import re
import unicodedata
from collections.abc import Iterable
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.load_data import Candidate

LOGGER = logging.getLogger(__name__)

_WHITESPACE_PATTERN = re.compile(r"\s+")
_CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SKILL_SEPARATOR_PATTERN = re.compile(r"[\s_\-/+.]+")
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")

_SKILL_ALIASES = {
    "amazon web services": "AWS",
    "aws": "AWS",
    "google cloud platform": "GCP",
    "gcp": "GCP",
    "microsoft azure": "Azure",
    "natural language processing": "NLP",
    "nlp": "NLP",
    "large language models": "LLMs",
    "large language model": "LLMs",
    "llm": "LLMs",
    "llms": "LLMs",
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "retrieval augmented generation": "RAG",
    "rag": "RAG",
    "parameter efficient fine tuning": "PEFT",
    "peft": "PEFT",
    "low rank adaptation": "LoRA",
    "lora": "LoRA",
    "qlora": "QLoRA",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "node js": "Node.js",
    "nodejs": "Node.js",
    "c sharp": "C#",
    "c plus plus": "C++",
    "natural language processing": "NLP",
    "information retrieval systems": "Information Retrieval",
    "vector representations": "Embeddings",
    "content matching": "Recommendation Systems",
    "workflow orchestration": "MLOps",
}

_DEFAULT_JOB_SKILLS = (
    "Python", "Machine Learning", "Deep Learning", "NLP",
    "Information Retrieval", "Recommendation Systems", "Embeddings",
    "Semantic Search", "Vector Search", "Hybrid Search", "BM25",
    "Sentence Transformers", "BGE", "E5", "Pinecone", "Weaviate",
    "Qdrant", "Milvus", "OpenSearch", "Elasticsearch", "FAISS",
    "pgvector", "RAG", "LLMs", "Fine-tuning LLMs", "LoRA", "QLoRA",
    "PEFT", "Learning to Rank", "XGBoost", "PyTorch", "MLOps",
    "A/B Testing", "NDCG", "MRR", "MAP", "Distributed Systems",
)

_REQUIREMENT_CUES = re.compile(
    r"\b(?:need|required|must|production|experience|strong|hands-on|"
    r"ranking|retrieval|search|embedding|evaluation|fine-tun|python|"
    r"vector|distributed|open-source|ideal candidate|first 90 days)\b",
    re.IGNORECASE,
)


def normalize_whitespace(text: str) -> str:
    """Collapse all Unicode whitespace into single ASCII spaces."""
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def clean_text(text: str | None) -> str:
    """Normalize Unicode, decode entities, and remove control characters."""
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text))
    decoded = html.unescape(normalized)
    without_controls = _CONTROL_CHARACTER_PATTERN.sub(" ", decoded)
    return normalize_whitespace(without_controls)


def _skill_key(skill_name: str) -> str:
    normalized = clean_text(skill_name).casefold()
    return normalize_whitespace(_SKILL_SEPARATOR_PATTERN.sub(" ", normalized))


def normalize_skill_names(skill_names: Iterable[str]) -> list[str]:
    """Canonicalize and de-duplicate skill names while preserving order."""
    normalized_names: list[str] = []
    seen: set[str] = set()
    for raw_name in skill_names:
        cleaned = clean_text(raw_name)
        if not cleaned:
            continue
        key = _skill_key(cleaned)
        canonical = _SKILL_ALIASES.get(key, cleaned)
        deduplication_key = _skill_key(canonical)
        if deduplication_key in seen:
            continue
        seen.add(deduplication_key)
        normalized_names.append(canonical)
    return normalized_names


def extract_skill_list(candidate: Candidate) -> list[str]:
    """Return a canonical, de-duplicated skill list for a candidate."""
    return normalize_skill_names(skill.name for skill in candidate.skills)


def extract_experience(candidate: Candidate) -> dict[str, float | int]:
    """Extract compact numerical experience features."""
    career_months = sum(role.duration_months for role in candidate.career_history)
    current_role_months = max(
        (
            role.duration_months
            for role in candidate.career_history
            if role.is_current
        ),
        default=0,
    )
    return {
        "years_of_experience": candidate.years_of_experience,
        "career_history_count": len(candidate.career_history),
        "career_duration_years": round(career_months / 12.0, 3),
        "current_role_months": current_role_months,
    }


def extract_behavior_features(candidate: Candidate) -> dict[str, Any]:
    """Flatten Redrob engagement signals into DataFrame-friendly features."""
    signals = candidate.redrob_signals
    assessment_scores = signals.skill_assessment_scores
    assessment_mean = (
        sum(assessment_scores.values()) / len(assessment_scores)
        if assessment_scores
        else 0.0
    )
    return {
        "profile_completeness_score": signals.profile_completeness_score,
        "signup_date": signals.signup_date,
        "last_active_date": signals.last_active_date,
        "open_to_work_flag": signals.open_to_work_flag,
        "profile_views_received_30d": signals.profile_views_received_30d,
        "applications_submitted_30d": signals.applications_submitted_30d,
        "recruiter_response_rate": signals.recruiter_response_rate,
        "avg_response_time_hours": signals.avg_response_time_hours,
        "skill_assessment_count": len(assessment_scores),
        "skill_assessment_mean": round(assessment_mean, 4),
        "skill_assessment_scores_json": json.dumps(
            assessment_scores,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "connection_count": signals.connection_count,
        "endorsements_received": signals.endorsements_received,
        "notice_period_days": signals.notice_period_days,
        "expected_salary_min_lpa": signals.expected_salary_min_lpa,
        "expected_salary_max_lpa": signals.expected_salary_max_lpa,
        "preferred_work_mode": signals.preferred_work_mode,
        "willing_to_relocate": signals.willing_to_relocate,
        "github_activity_score": signals.github_activity_score,
        "search_appearance_30d": signals.search_appearance_30d,
        "saved_by_recruiters_30d": signals.saved_by_recruiters_30d,
        "interview_completion_rate": signals.interview_completion_rate,
        "offer_acceptance_rate": signals.offer_acceptance_rate,
        "verified_email": signals.verified_email,
        "verified_phone": signals.verified_phone,
        "linkedin_connected": signals.linkedin_connected,
    }


def extract_job_requirements(
    job_description: str,
    *,
    maximum: int = 24,
) -> list[str]:
    """Extract concise requirement-bearing sentences from a job description."""
    if maximum < 1:
        raise ValueError("maximum must be positive")
    normalized = unicodedata.normalize("NFKC", job_description)
    candidates = [clean_text(part) for part in _SENTENCE_PATTERN.split(normalized)]
    selected: list[str] = []
    seen: set[str] = set()
    for sentence in candidates:
        if not sentence or len(sentence) < 20:
            continue
        if not _REQUIREMENT_CUES.search(sentence):
            continue
        key = sentence.casefold()
        if key in seen:
            continue
        seen.add(key)
        selected.append(sentence)
        if len(selected) == maximum:
            break
    if not selected:
        selected.append(preprocess_job_description(job_description)[:1_500])
    return selected


def extract_job_skills(
    job_description: str,
    skill_vocabulary: Iterable[str] | None = None,
) -> list[str]:
    """Extract skills explicitly mentioned in a JD from a known vocabulary."""
    vocabulary = normalize_skill_names(
        tuple(_DEFAULT_JOB_SKILLS) + tuple(skill_vocabulary or ())
    )
    searchable = f" {clean_text(job_description).casefold()} "
    matches: list[str] = []
    for skill in vocabulary:
        variants = {_skill_key(skill), clean_text(skill).casefold()}
        variants.update(
            alias for alias, canonical in _SKILL_ALIASES.items()
            if _skill_key(canonical) == _skill_key(skill)
        )
        patterns = [
            re.escape(variant).replace(r"\ ", r"[\s\-_/]+")
            for variant in variants
            if variant
        ]
        if any(
            re.search(rf"(?<![\w]){pattern}(?![\w])", searchable)
            for pattern in patterns
        ):
            matches.append(skill)
    return normalize_skill_names(matches)


def build_candidate_text(candidate: Candidate) -> str:
    """Build labeled semantic text from all relevant candidate evidence."""
    skill_names = extract_skill_list(candidate)
    career_entries = [
        clean_text(
            f"{role.title} at {role.company}. {role.description}"
        )
        for role in candidate.career_history
    ]
    education_entries = [
        clean_text(
            f"{item.degree} in {item.field_of_study}, "
            f"{item.institution} ({item.start_year}-{item.end_year})"
        )
        for item in candidate.education
    ]
    certification_entries = [
        clean_text(f"{item.name}, {item.issuer} ({item.year})")
        for item in candidate.certifications
    ]

    sections = [
        ("Headline", clean_text(candidate.headline)),
        ("Summary", clean_text(candidate.summary)),
        ("Current title", clean_text(candidate.current_title)),
        ("Career history", " ".join(filter(None, career_entries))),
        ("Skills", ", ".join(skill_names)),
        ("Education", "; ".join(filter(None, education_entries))),
        (
            "Certifications",
            "; ".join(filter(None, certification_entries)),
        ),
    ]
    return "\n".join(
        f"{label}: {value}" for label, value in sections if value
    )


def _candidate_record(candidate: Candidate) -> dict[str, Any]:
    skill_names = extract_skill_list(candidate)
    record: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "candidate_text": build_candidate_text(candidate),
        "headline": clean_text(candidate.headline),
        "current_title": clean_text(candidate.current_title),
        "current_company": clean_text(candidate.current_company),
        "current_company_size": candidate.current_company_size,
        "current_industry": clean_text(candidate.current_industry),
        "location": clean_text(candidate.location),
        "country": clean_text(candidate.country),
        "skill_names_json": json.dumps(skill_names, ensure_ascii=False),
        "career_titles_json": json.dumps(
            [clean_text(role.title) for role in candidate.career_history],
            ensure_ascii=False,
        ),
        "career_companies_json": json.dumps(
            [clean_text(role.company) for role in candidate.career_history],
            ensure_ascii=False,
        ),
        "career_history_json": json.dumps(
            [
                {
                    "company": clean_text(role.company),
                    "title": clean_text(role.title),
                    "start_date": role.start_date,
                    "end_date": role.end_date,
                    "duration_months": role.duration_months,
                    "is_current": role.is_current,
                }
                for role in candidate.career_history
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "skill_details_json": json.dumps(
            [
                {
                    "name": normalize_skill_names([skill.name])[0],
                    "proficiency": skill.proficiency,
                    "endorsements": skill.endorsements,
                    "duration_months": skill.duration_months,
                }
                for skill in candidate.skills
                if clean_text(skill.name)
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "skill_count": len(skill_names),
        "education_count": len(candidate.education),
        "certification_count": len(candidate.certifications),
        "language_count": len(candidate.languages),
    }
    record.update(extract_experience(candidate))
    record.update(extract_behavior_features(candidate))
    return record


def extract_candidate_dataframe(
    candidates: Iterable[Candidate],
    *,
    show_progress: bool = True,
    total: int | None = None,
) -> pd.DataFrame:
    """Convert a candidate stream into an embedding-ready DataFrame."""
    iterator = tqdm(
        candidates,
        total=total,
        desc="Preprocessing candidates",
        unit="candidate",
        disable=not show_progress,
        dynamic_ncols=True,
    )
    records = [_candidate_record(candidate) for candidate in iterator]
    dataframe = pd.DataFrame.from_records(records)
    if dataframe.empty:
        LOGGER.warning("Candidate stream produced an empty DataFrame")
        return dataframe

    string_columns = [
        "candidate_id",
        "candidate_text",
        "headline",
        "current_title",
        "current_company",
        "current_company_size",
        "current_industry",
        "location",
        "country",
        "signup_date",
        "last_active_date",
        "preferred_work_mode",
        "skill_assessment_scores_json",
        "skill_names_json",
        "career_titles_json",
        "career_companies_json",
        "career_history_json",
        "skill_details_json",
    ]
    dataframe[string_columns] = dataframe[string_columns].astype("string")

    integer_columns = [
        "skill_count",
        "education_count",
        "certification_count",
        "language_count",
        "career_history_count",
        "current_role_months",
        "profile_views_received_30d",
        "applications_submitted_30d",
        "skill_assessment_count",
        "connection_count",
        "endorsements_received",
        "notice_period_days",
        "search_appearance_30d",
        "saved_by_recruiters_30d",
    ]
    dataframe[integer_columns] = dataframe[integer_columns].astype("int32")

    boolean_columns = [
        "open_to_work_flag",
        "willing_to_relocate",
        "verified_email",
        "verified_phone",
        "linkedin_connected",
    ]
    dataframe[boolean_columns] = dataframe[boolean_columns].astype("bool")
    return dataframe


def preprocess_job_description(job_description: str) -> str:
    """Normalize a job description for later semantic embedding."""
    processed = clean_text(job_description)
    if not processed:
        raise ValueError("Job description is empty after preprocessing")
    return processed
