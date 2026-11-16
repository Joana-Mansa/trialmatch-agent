"""Deterministic eligibility checks done in code, not by the model.

Age limits and sex are structured fields on ClinicalTrials.gov. Comparing them is
arithmetic, so code does it and the agent is told the result. The grounding step later
flags any report that calls a patient eligible when one of these checks failed.

Note: CT.gov's ``healthyVolunteers`` flag means the trial *accepts* healthy volunteers,
not that it requires them, so it is not a disqualifier and is not checked here.
"""

from __future__ import annotations

import re

from .models import HardCheckResult, PatientSummary, TrialEligibility

_AGE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(year|month|week|day|hour|minute)s?\s*$", re.I)
_DAYS_PER = {"year": 365.25, "month": 30.4375, "week": 7.0, "day": 1.0, "hour": 1 / 24,
             "minute": 1 / 1440}


def parse_age(raw: str | None) -> tuple[float, str] | None:
    """Parse a CT.gov age string like "18 Years" into (value, unit). None for N/A or blank."""
    if raw is None:
        return None
    m = _AGE_RE.match(raw)
    if not m:
        return None
    return float(m.group(1)), m.group(2).lower()


def _patient_age_days(patient: PatientSummary) -> int:
    return (patient.as_of - patient.birth_date).days


def check(patient: PatientSummary, elig: TrialEligibility) -> HardCheckResult:
    checks: dict[str, str] = {}
    notes: list[str] = []

    for key, raw, is_min in (
        ("minimum_age", elig.minimum_age, True),
        ("maximum_age", elig.maximum_age, False),
    ):
        parsed = parse_age(raw)
        if parsed is None:
            checks[key] = "not_applicable"
            if raw not in (None, "", "N/A"):
                notes.append(f"Could not parse {key} {raw!r}; not checked.")
            continue
        value, unit = parsed
        if unit == "year":
            # Whole years, inclusive bounds: "75 Years" admits a 75-year-old.
            ok = patient.age_years >= value if is_min else patient.age_years <= value
        else:
            limit_days = value * _DAYS_PER[unit]
            age_days = _patient_age_days(patient)
            ok = age_days >= limit_days if is_min else age_days <= limit_days
        checks[key] = "pass" if ok else "fail"
        if not ok:
            bound = "below the minimum" if is_min else "above the maximum"
            notes.append(f"Patient age {patient.age_years} is {bound} age {raw}.")

    if elig.sex == "ALL":
        checks["sex"] = "pass"
    elif patient.sex in ("male", "female"):
        ok = patient.sex.upper() == elig.sex
        checks["sex"] = "pass" if ok else "fail"
        if not ok:
            notes.append(f"Trial enrols {elig.sex} only; patient sex is {patient.sex}.")
    else:
        checks["sex"] = "not_applicable"
        notes.append(f"Trial enrols {elig.sex} only; patient sex is {patient.sex}, not checked.")

    return HardCheckResult(
        nct_id=elig.nct_id,
        passed=all(v != "fail" for v in checks.values()),
        checks=checks,  # type: ignore[arg-type]
        notes=notes,
    )
