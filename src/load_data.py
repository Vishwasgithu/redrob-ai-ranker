"""Typed, streaming ingestion for the Redrob candidate dataset."""

from __future__ import annotations

import json
import logging
import zipfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

JsonObject = dict[str, Any]
SchemaValidator = Callable[[Mapping[str, Any]], Any]


class DataIngestionError(RuntimeError):
    """Base exception for dataset ingestion failures."""


class CandidateValidationError(DataIngestionError):
    """Raised when a candidate does not satisfy the supplied JSON Schema."""


class CandidateDeserializationError(DataIngestionError):
    """Raised when a candidate cannot be converted to typed data classes."""


@dataclass(frozen=True, slots=True)
class Skill:
    """A candidate skill and its evidence fields."""

    name: str
    proficiency: str
    endorsements: int
    duration_months: int | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Skill:
        """Build a skill from a validated JSON mapping."""
        duration = value.get("duration_months")
        return cls(
            name=str(value["name"]),
            proficiency=str(value["proficiency"]),
            endorsements=int(value["endorsements"]),
            duration_months=int(duration) if duration is not None else None,
        )


@dataclass(frozen=True, slots=True)
class Career:
    """One position in a candidate's employment history."""

    company: str
    title: str
    start_date: str
    end_date: str | None
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Career:
        """Build a career entry from a validated JSON mapping."""
        end_date = value.get("end_date")
        return cls(
            company=str(value["company"]),
            title=str(value["title"]),
            start_date=str(value["start_date"]),
            end_date=str(end_date) if end_date is not None else None,
            duration_months=int(value["duration_months"]),
            is_current=bool(value["is_current"]),
            industry=str(value["industry"]),
            company_size=str(value["company_size"]),
            description=str(value["description"]),
        )


@dataclass(frozen=True, slots=True)
class Education:
    """One education record for a candidate."""

    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: str | None = None
    tier: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Education:
        """Build an education entry from a validated JSON mapping."""
        grade = value.get("grade")
        tier = value.get("tier")
        return cls(
            institution=str(value["institution"]),
            degree=str(value["degree"]),
            field_of_study=str(value["field_of_study"]),
            start_year=int(value["start_year"]),
            end_year=int(value["end_year"]),
            grade=str(grade) if grade is not None else None,
            tier=str(tier) if tier is not None else None,
        )


@dataclass(frozen=True, slots=True)
class Certification:
    """A professional certification listed by a candidate."""

    name: str
    issuer: str
    year: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Certification:
        """Build a certification from a validated JSON mapping."""
        return cls(
            name=str(value["name"]),
            issuer=str(value["issuer"]),
            year=int(value["year"]),
        )


@dataclass(frozen=True, slots=True)
class Language:
    """A spoken language and proficiency level."""

    language: str
    proficiency: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Language:
        """Build a language from a validated JSON mapping."""
        return cls(
            language=str(value["language"]),
            proficiency=str(value["proficiency"]),
        )


@dataclass(frozen=True, slots=True)
class RedrobSignals:
    """Behavioral and engagement signals supplied by Redrob."""

    profile_completeness_score: float
    signup_date: str
    last_active_date: str
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: float
    avg_response_time_hours: float
    skill_assessment_scores: dict[str, float]
    connection_count: int
    endorsements_received: int
    notice_period_days: int
    expected_salary_min_lpa: float
    expected_salary_max_lpa: float
    preferred_work_mode: str
    willing_to_relocate: bool
    github_activity_score: float
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: float
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RedrobSignals:
        """Build behavioral signals from a validated JSON mapping."""
        salary = _require_mapping(
            value.get("expected_salary_range_inr_lpa"),
            "redrob_signals.expected_salary_range_inr_lpa",
        )
        assessments = _require_mapping(
            value.get("skill_assessment_scores"),
            "redrob_signals.skill_assessment_scores",
        )
        return cls(
            profile_completeness_score=float(
                value["profile_completeness_score"]
            ),
            signup_date=str(value["signup_date"]),
            last_active_date=str(value["last_active_date"]),
            open_to_work_flag=bool(value["open_to_work_flag"]),
            profile_views_received_30d=int(
                value["profile_views_received_30d"]
            ),
            applications_submitted_30d=int(
                value["applications_submitted_30d"]
            ),
            recruiter_response_rate=float(value["recruiter_response_rate"]),
            avg_response_time_hours=float(value["avg_response_time_hours"]),
            skill_assessment_scores={
                str(name): float(score)
                for name, score in assessments.items()
            },
            connection_count=int(value["connection_count"]),
            endorsements_received=int(value["endorsements_received"]),
            notice_period_days=int(value["notice_period_days"]),
            expected_salary_min_lpa=float(salary["min"]),
            expected_salary_max_lpa=float(salary["max"]),
            preferred_work_mode=str(value["preferred_work_mode"]),
            willing_to_relocate=bool(value["willing_to_relocate"]),
            github_activity_score=float(value["github_activity_score"]),
            search_appearance_30d=int(value["search_appearance_30d"]),
            saved_by_recruiters_30d=int(value["saved_by_recruiters_30d"]),
            interview_completion_rate=float(
                value["interview_completion_rate"]
            ),
            offer_acceptance_rate=float(value["offer_acceptance_rate"]),
            verified_email=bool(value["verified_email"]),
            verified_phone=bool(value["verified_phone"]),
            linkedin_connected=bool(value["linkedin_connected"]),
        )


@dataclass(frozen=True, slots=True)
class Candidate:
    """Typed candidate profile used by downstream preprocessing."""

    candidate_id: str
    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str
    career_history: tuple[Career, ...]
    education: tuple[Education, ...]
    skills: tuple[Skill, ...]
    certifications: tuple[Certification, ...]
    languages: tuple[Language, ...]
    redrob_signals: RedrobSignals

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Candidate:
        """Create a candidate without retaining the large raw JSON mapping."""
        try:
            profile = _require_mapping(value.get("profile"), "profile")
            career_history = _require_sequence(
                value.get("career_history"), "career_history"
            )
            education = _require_sequence(value.get("education"), "education")
            skills = _require_sequence(value.get("skills"), "skills")
            certifications = _require_sequence(
                value.get("certifications", []), "certifications"
            )
            languages = _require_sequence(
                value.get("languages", []), "languages"
            )
            signals = _require_mapping(
                value.get("redrob_signals"), "redrob_signals"
            )
            return cls(
                candidate_id=str(value["candidate_id"]),
                anonymized_name=str(profile["anonymized_name"]),
                headline=str(profile["headline"]),
                summary=str(profile["summary"]),
                location=str(profile["location"]),
                country=str(profile["country"]),
                years_of_experience=float(profile["years_of_experience"]),
                current_title=str(profile["current_title"]),
                current_company=str(profile["current_company"]),
                current_company_size=str(profile["current_company_size"]),
                current_industry=str(profile["current_industry"]),
                career_history=tuple(
                    Career.from_dict(_require_mapping(item, "career_history[]"))
                    for item in career_history
                ),
                education=tuple(
                    Education.from_dict(_require_mapping(item, "education[]"))
                    for item in education
                ),
                skills=tuple(
                    Skill.from_dict(_require_mapping(item, "skills[]"))
                    for item in skills
                ),
                certifications=tuple(
                    Certification.from_dict(
                        _require_mapping(item, "certifications[]")
                    )
                    for item in certifications
                ),
                languages=tuple(
                    Language.from_dict(_require_mapping(item, "languages[]"))
                    for item in languages
                ),
                redrob_signals=RedrobSignals.from_dict(signals),
            )
        except (KeyError, TypeError, ValueError) as exc:
            candidate_id = value.get("candidate_id", "<unknown>")
            raise CandidateDeserializationError(
                f"Unable to deserialize candidate {candidate_id}: {exc}"
            ) from exc


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    return value


def _require_sequence(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be an array")
    return value


def _compile_validator(schema: Mapping[str, Any]) -> SchemaValidator:
    """Compile the schema once, preferring the optimized validator."""
    try:
        import fastjsonschema

        return fastjsonschema.compile(dict(schema))
    except ImportError:
        LOGGER.warning(
            "fastjsonschema is unavailable; falling back to jsonschema. "
            "Install requirements.txt for faster full-dataset validation."
        )

    try:
        from jsonschema.validators import validator_for
    except ImportError as exc:
        raise DataIngestionError(
            "Candidate validation requires fastjsonschema or jsonschema."
        ) from exc

    validator_class = validator_for(schema)
    validator_class.check_schema(schema)
    validator = validator_class(schema)

    def validate_with_jsonschema(candidate: Mapping[str, Any]) -> None:
        validator.validate(candidate)

    return validate_with_jsonschema


def read_jsonl(path: str | Path) -> Iterator[JsonObject]:
    """Yield JSON objects from a JSONL file without loading it into memory."""
    jsonl_path = Path(path).expanduser().resolve()
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    try:
        with jsonl_path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataIngestionError(
                        f"Invalid JSON at {jsonl_path}:{line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(value, dict):
                    raise DataIngestionError(
                        f"Expected a JSON object at "
                        f"{jsonl_path}:{line_number}, got "
                        f"{type(value).__name__}."
                    )
                yield value
    except UnicodeDecodeError as exc:
        raise DataIngestionError(
            f"JSONL file is not valid UTF-8: {jsonl_path}"
        ) from exc
    except OSError as exc:
        raise DataIngestionError(f"Unable to read {jsonl_path}: {exc}") from exc


def validate_candidate(
    candidate: Mapping[str, Any],
    schema: Mapping[str, Any] | None = None,
    *,
    validator: SchemaValidator | None = None,
    raise_on_error: bool = True,
) -> bool:
    """Validate one candidate against the supplied or compiled schema.

    A compiled validator should be passed when validating many candidates.
    This avoids rebuilding the schema validator for every JSONL record.
    """
    if validator is None:
        if schema is None:
            raise ValueError("schema or validator must be provided")
        validator = _compile_validator(schema)

    try:
        validator(candidate)
    except Exception as exc:
        candidate_id = candidate.get("candidate_id", "<unknown>")
        if raise_on_error:
            raise CandidateValidationError(
                f"Candidate {candidate_id} failed schema validation: {exc}"
            ) from exc
        LOGGER.debug(
            "Candidate %s failed schema validation: %s", candidate_id, exc
        )
        return False
    return True


def stream_candidates(
    path: str | Path,
    *,
    schema: Mapping[str, Any] | None = None,
    validate: bool = True,
    show_progress: bool = True,
    total: int | None = None,
) -> Iterator[Candidate]:
    """Stream validated, typed candidates from a JSONL file."""
    jsonl_path = Path(path).expanduser().resolve()
    active_schema = schema
    if validate and active_schema is None:
        active_schema = load_schema(jsonl_path.with_name("candidate_schema.json"))
    validator = (
        _compile_validator(active_schema)
        if validate and active_schema is not None
        else None
    )
    raw_candidates = read_jsonl(jsonl_path)
    progress = tqdm(
        raw_candidates,
        total=total,
        desc="Loading candidates",
        unit="candidate",
        disable=not show_progress,
        dynamic_ncols=True,
    )

    for record_number, raw_candidate in enumerate(progress, start=1):
        try:
            if validator is not None:
                validate_candidate(raw_candidate, validator=validator)
            yield Candidate.from_dict(raw_candidate)
        except DataIngestionError:
            LOGGER.exception("Candidate record %d could not be loaded", record_number)
            raise


def load_candidates(
    path: str | Path,
    *,
    schema: Mapping[str, Any] | None = None,
    validate: bool = True,
    show_progress: bool = True,
    total: int | None = None,
) -> list[Candidate]:
    """Load all candidates into memory.

    Prefer :func:`stream_candidates` for the full dataset. This convenience
    function is useful for tests and smaller candidate subsets.
    """
    return list(
        stream_candidates(
            path,
            schema=schema,
            validate=validate,
            show_progress=show_progress,
            total=total,
        )
    )


def load_schema(path: str | Path) -> JsonObject:
    """Load and minimally verify a JSON Schema document."""
    schema_path = Path(path).expanduser().resolve()
    if not schema_path.is_file():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataIngestionError(
            f"Invalid JSON Schema at {schema_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise DataIngestionError(
            f"Unable to read schema {schema_path}: {exc}"
        ) from exc
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise DataIngestionError(
            f"Schema root must be a JSON object schema: {schema_path}"
        )
    return schema


def _paragraph_text(element: ElementTree.Element, namespace: str) -> str:
    """Extract visible text from one WordprocessingML paragraph/cell."""
    parts: list[str] = []
    for child in element.iter():
        if child.tag == f"{{{namespace}}}t" and child.text:
            parts.append(child.text)
        elif child.tag == f"{{{namespace}}}tab":
            parts.append("\t")
        elif child.tag in {f"{{{namespace}}}br", f"{{{namespace}}}cr"}:
            parts.append("\n")
    return "".join(parts).strip()


def read_docx(path: str | Path) -> str:
    """Extract body text from DOCX paragraphs and tables in document order."""
    docx_path = Path(path).expanduser().resolve()
    if not docx_path.is_file():
        raise FileNotFoundError(f"DOCX file not found: {docx_path}")

    namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    paragraph_tag = f"{{{namespace}}}p"
    table_tag = f"{{{namespace}}}tbl"
    row_tag = f"{{{namespace}}}tr"
    cell_tag = f"{{{namespace}}}tc"

    try:
        with zipfile.ZipFile(docx_path) as archive:
            document_xml = archive.read("word/document.xml")
        root = ElementTree.fromstring(document_xml)
    except (KeyError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
        raise DataIngestionError(
            f"Invalid or unsupported DOCX file {docx_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise DataIngestionError(f"Unable to read {docx_path}: {exc}") from exc

    body = root.find(f"{{{namespace}}}body")
    if body is None:
        raise DataIngestionError(f"DOCX has no document body: {docx_path}")

    blocks: list[str] = []
    for block in body:
        if block.tag == paragraph_tag:
            text = _paragraph_text(block, namespace)
            if text:
                blocks.append(text)
        elif block.tag == table_tag:
            for row in block.findall(f"./{row_tag}"):
                cells = [
                    _paragraph_text(cell, namespace)
                    for cell in row.findall(f"./{cell_tag}")
                ]
                row_text = " | ".join(cell for cell in cells if cell)
                if row_text:
                    blocks.append(row_text)
    return "\n".join(blocks)


def load_job_description(path: str | Path) -> str:
    """Load a non-empty job description from DOCX or UTF-8 text."""
    job_path = Path(path).expanduser().resolve()
    if job_path.suffix.casefold() == ".docx":
        text = read_docx(job_path)
    else:
        try:
            text = job_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DataIngestionError(
                f"Unable to read job description {job_path}: {exc}"
            ) from exc
    if not text.strip():
        raise DataIngestionError(f"Job description is empty: {job_path}")
    return text
