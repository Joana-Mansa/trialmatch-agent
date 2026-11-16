"""Shared data contracts.

Every module codes against these types. Change them only with care: the patient
store, the trials client, the agent, the eval harness and the API all depend on them.

Anything taken from a patient record carries ``resource_id`` (the FHIR resource id,
e.g. ``"Observation/obs-egfr-1"``) so the agent can cite it and code can verify it.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- patient


class Condition(BaseModel):
    resource_id: str
    code: str | None = None  # SNOMED CT or ICD-10 code when available
    display: str
    clinical_status: str | None = None  # active, resolved, remission, ...
    onset: date | None = None


class Medication(BaseModel):
    resource_id: str
    code: str | None = None  # RxNorm when available
    display: str
    status: str | None = None  # active, completed, stopped, ...
    authored_on: date | None = None


class LabResult(BaseModel):
    resource_id: str
    code: str | None = None  # LOINC when available
    display: str
    value: float | str | None = None
    comparator: str | None = None  # FHIR valueQuantity.comparator: "<", "<=", ">=", ">"
    unit: str | None = None
    effective: date | None = None


class Procedure(BaseModel):
    resource_id: str
    code: str | None = None
    display: str
    performed: date | None = None


class PatientSummary(BaseModel):
    patient_id: str
    resource_id: str  # "Patient/<id>"
    sex: Literal["male", "female", "other", "unknown"]
    birth_date: date
    age_years: int  # computed in code against ``as_of``, never by the model
    as_of: date
    country: str | None = None
    conditions: list[Condition] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    latest_labs: list[LabResult] = Field(default_factory=list)  # most recent per code
    procedures: list[Procedure] = Field(default_factory=list)
    pregnancy_status: str | None = None  # from an Observation when recorded
    pregnancy_status_ref: str | None = None  # resource id of that Observation, citable
    pregnancy_status_date: date | None = None  # when it was recorded; old values may be stale


# --------------------------------------------------------------------------- trials


class TrialHit(BaseModel):
    nct_id: str
    title: str
    phases: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    status: str


class TrialEligibility(BaseModel):
    nct_id: str
    title: str
    status: str = "UNKNOWN"  # CT.gov overallStatus, e.g. RECRUITING, TERMINATED
    criteria_text: str
    sex: Literal["ALL", "MALE", "FEMALE"] = "ALL"
    minimum_age: str | None = None  # raw CT.gov string, e.g. "18 Years"
    maximum_age: str | None = None
    healthy_volunteers: bool | None = None
    countries: list[str] = Field(default_factory=list)


class HardCheckResult(BaseModel):
    """Deterministic checks computed in code (age, sex, healthy volunteers)."""

    nct_id: str
    passed: bool  # False if any individual check failed
    checks: dict[str, Literal["pass", "fail", "not_applicable"]]
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- report

CriterionStatus = Literal["met", "not_met", "unknown"]
Verdict = Literal["likely_eligible", "possibly_eligible", "likely_ineligible"]


class LabCheck(BaseModel):
    """A numeric criterion stated so code can verify it: "<lab> <operator> <value> <unit>".

    The condition is the criterion as written, for either kind: for an exclusion
    "eGFR < 30", the condition is eGFR < 30, and the exclusion is met when it holds.
    """

    lab_ref: str  # the Observation cited for this criterion
    operator: Literal["<", "<=", ">", ">="]
    value: float
    unit: str | None = None


class CriterionAssessment(BaseModel):
    criterion: str  # one criterion, paraphrased or quoted from the trial text
    kind: Literal["inclusion", "exclusion"]
    # For inclusion: "met" means the patient satisfies it.
    # For exclusion: "met" means the exclusion applies (patient is excluded by it).
    status: CriterionStatus
    evidence_refs: list[str] = Field(default_factory=list)  # FHIR resource ids
    rationale: str
    lab_check: LabCheck | None = None  # set for numeric lab thresholds; code verifies it


class TrialMatch(BaseModel):
    nct_id: str
    title: str
    verdict: Verdict
    rank: int  # 1 = best match
    summary: str
    assessments: list[CriterionAssessment]
    missing_info: list[str] = Field(default_factory=list)


class MatchReport(BaseModel):
    patient_id: str
    matches: list[TrialMatch]
    trials_considered: int
    notes: str = ""


# --------------------------------------------------------------------------- run


class TraceStep(BaseModel):
    step: int
    tool: str
    input: dict
    ok: bool
    duration_ms: float
    error: str | None = None


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


class RunTrace(BaseModel):
    run_id: str
    patient_id: str
    model: str
    turns: int = 0
    steps: list[TraceStep] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    final_stop_reason: str | None = None
    fallback_used: bool = False
    wall_seconds: float = 0.0
    request_ids: list[str] = Field(default_factory=list)


class GroundingIssue(BaseModel):
    kind: Literal[
        "unknown_evidence_ref",  # cited a resource id not in the patient record
        "hard_check_contradiction",  # verdict eligible but a code check failed
        "unknown_trial",  # nct_id never retrieved during the run
        "empty_assessments",
        "not_recruiting",  # trial is not RECRUITING
        "verdict_inconsistent",  # verdict contradicts the criterion assessments
        "duplicate_trial",  # same nct_id or rank appears twice
        "criteria_truncated",  # model saw only part of the criteria text
        "lab_value_contradiction",  # status disagrees with the cited lab value
        "lab_check_invalid",  # lab_check cannot be evaluated as stated
        "evidence_mismatch",  # cited medication is outside the drug class named
    ]
    nct_id: str | None = None
    detail: str


class MatchResult(BaseModel):
    """What the agent returns: the report, how it was produced, and what code verified."""

    # completed_with_issues: a report was produced but code-level grounding found problems
    # the model did not fix; treat it as needing human review.
    status: Literal["completed", "completed_with_issues", "refused", "incomplete", "error"]
    report: MatchReport | None = None
    trace: RunTrace
    grounding_issues: list[GroundingIssue] = Field(default_factory=list)
    error: str | None = None
