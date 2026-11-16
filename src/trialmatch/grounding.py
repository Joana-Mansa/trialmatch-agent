"""Code checks on a submitted report: is everything it cites real and consistent?

These checks do not judge clinical reasoning. They catch the mechanical failures a
model can make: citing a resource id that is not in the record, reporting a trial it
never retrieved or that is not recruiting, ignoring a failed hard check, giving a
verdict that contradicts its own criterion assessments, or calling a trial likely
eligible when part of its criteria text was never seen.

Two checks look at what a citation says, not only that it exists: a stated lab
threshold is recomputed from the cited value, and a medication cited for a drug-class
criterion must belong to that class (``drug_classes.py``).

The agent runs these checks when the report is submitted and sends any problems back
to the model to fix, then runs them again on the final report.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable, Mapping

from .drug_classes import unsupported_medication_claim
from .models import (
    CriterionAssessment,
    GroundingIssue,
    HardCheckResult,
    LabResult,
    MatchReport,
    TrialEligibility,
    TrialMatch,
)
from .ports import PatientNotFound, PatientStore

TRUNCATION_MARKER = "[truncated:"

_OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}


def _norm_unit(unit: str | None) -> str:
    return "".join((unit or "").lower().split())


class _Record:
    """The patient's medications and lab results by resource id, for semantic checks."""

    def __init__(self, store: PatientStore, patient_id: str):
        self.medications: dict[str, str] = {}
        self.labs: dict[str, LabResult] = {}
        try:
            summary = store.summary(patient_id)
            labs = store.labs(patient_id, "")
        except PatientNotFound:
            return
        self.medications = {m.resource_id: m.display for m in summary.medications}
        self.labs = {lab.resource_id: lab for lab in [*summary.latest_labs, *labs]}


def _lab_check_issue(a: CriterionAssessment, nct: str, record: _Record) -> GroundingIssue | None:
    """Recompute a stated lab threshold from the cited value, like the age check."""
    chk = a.lab_check
    if chk is None:
        return None
    where = f"criterion {a.criterion[:100]!r}"

    def invalid(detail: str) -> GroundingIssue:
        return GroundingIssue(kind="lab_check_invalid", nct_id=nct, detail=f"{where}: {detail}")

    if chk.lab_ref not in a.evidence_refs:
        return invalid(f"lab_check.lab_ref {chk.lab_ref!r} must also be in evidence_refs.")
    lab = record.labs.get(chk.lab_ref)
    if lab is None:
        return invalid(f"lab_check.lab_ref {chk.lab_ref!r} is not a lab result in the record.")
    if not isinstance(lab.value, (int, float)) or isinstance(lab.value, bool):
        return invalid(f"{chk.lab_ref} has no numeric value; set lab_check to null.")
    if lab.comparator:
        return None  # e.g. "<5": the exact value is unknown, so code does not decide
    if chk.unit and lab.unit and _norm_unit(chk.unit) != _norm_unit(lab.unit):
        return invalid(f"threshold unit {chk.unit!r} differs from the record's unit "
                       f"{lab.unit!r}; convert the threshold to {lab.unit!r}.")
    if a.status == "unknown":
        return None  # allowed, e.g. when the value is too old for the criterion
    holds = _OPS[chk.operator](float(lab.value), chk.value)
    expected = "met" if holds else "not_met"
    if a.status == expected:
        return None
    unit = f" {lab.unit}" if lab.unit else ""
    return GroundingIssue(
        kind="lab_value_contradiction",
        nct_id=nct,
        detail=f"{where}: {lab.display} is {lab.value:g}{unit} ({chk.lab_ref}), and "
        f"{lab.value:g} {chk.operator} {chk.value:g} is {holds}, so status must be "
        f"{expected}, not {a.status}.",
    )


def _medication_issue(
    a: CriterionAssessment, nct: str, record: _Record
) -> GroundingIssue | None:
    if a.status != "met":
        return None
    meds = [r for r in a.evidence_refs if r.startswith(("MedicationRequest/",
                                                       "MedicationStatement/"))]
    if not meds or any(r not in record.medications for r in meds):
        return None
    why = unsupported_medication_claim(a.criterion, [record.medications[r] for r in meds])
    if why is None:
        return None
    return GroundingIssue(
        kind="evidence_mismatch",
        nct_id=nct,
        detail=f"criterion {a.criterion[:100]!r} is marked met but {why}. Cite a "
        "medication in that class, or change the status (unknown if the record does not "
        "settle it).",
    )


def _verdict_issues(match: TrialMatch) -> list[GroundingIssue]:
    """Enforce the verdict rules from the system prompt in code."""
    blocking = [
        a
        for a in match.assessments
        if (a.kind == "exclusion" and a.status == "met")
        or (a.kind == "inclusion" and a.status == "not_met")
    ]
    out: list[GroundingIssue] = []
    if blocking and match.verdict != "likely_ineligible":
        first = blocking[0]
        out.append(
            GroundingIssue(
                kind="verdict_inconsistent",
                nct_id=match.nct_id,
                detail=f"Verdict is {match.verdict} but {len(blocking)} assessment(s) block "
                f"eligibility (e.g. {first.kind}: {first.criterion[:100]!r}). Blocking items "
                "require likely_ineligible.",
            )
        )
    if match.verdict == "likely_eligible":
        if not any(a.evidence_refs for a in match.assessments):
            out.append(
                GroundingIssue(
                    kind="verdict_inconsistent",
                    nct_id=match.nct_id,
                    detail="Verdict is likely_eligible but no assessment cites any evidence.",
                )
            )
        if not any(a.kind == "inclusion" and a.status == "met" for a in match.assessments):
            out.append(
                GroundingIssue(
                    kind="verdict_inconsistent",
                    nct_id=match.nct_id,
                    detail="Verdict is likely_eligible but no inclusion criterion is met.",
                )
            )
    return out


def validate(
    report: MatchReport,
    store: PatientStore,
    seen_nct_ids: Iterable[str],
    hard_results: Mapping[str, HardCheckResult],
    eligibility: Mapping[str, TrialEligibility] | None = None,
) -> list[GroundingIssue]:
    seen = set(seen_nct_ids)
    elig = eligibility or {}
    issues: list[GroundingIssue] = []
    record = _Record(store, report.patient_id)

    ids = [m.nct_id for m in report.matches]
    for nct in sorted({n for n in ids if ids.count(n) > 1}):
        issues.append(
            GroundingIssue(kind="duplicate_trial", nct_id=nct, detail=f"{nct} appears twice.")
        )
    ranks = [m.rank for m in report.matches]
    for rank in sorted({r for r in ranks if ranks.count(r) > 1}):
        issues.append(
            GroundingIssue(
                kind="duplicate_trial", detail=f"Rank {rank} is used by more than one match."
            )
        )

    for match in report.matches:
        nct = match.nct_id

        if nct not in seen:
            issues.append(
                GroundingIssue(
                    kind="unknown_trial",
                    nct_id=nct,
                    detail=f"{nct} was never returned by search or eligibility lookup.",
                )
            )

        if not match.assessments:
            issues.append(
                GroundingIssue(
                    kind="empty_assessments",
                    nct_id=nct,
                    detail="Match has no criterion-level assessments.",
                )
            )

        issues.extend(_verdict_issues(match))

        e = elig.get(nct)
        if e is not None and e.status not in ("RECRUITING", "UNKNOWN"):
            issues.append(
                GroundingIssue(
                    kind="not_recruiting",
                    nct_id=nct,
                    detail=f"{nct} has status {e.status}; only recruiting trials are candidates.",
                )
            )
        if e is not None and TRUNCATION_MARKER in e.criteria_text and (
            match.verdict == "likely_eligible"
        ):
            issues.append(
                GroundingIssue(
                    kind="criteria_truncated",
                    nct_id=nct,
                    detail="Criteria text was truncated, so not every criterion was seen; "
                    "likely_eligible is not supported. Use possibly_eligible at most.",
                )
            )

        checked: set[str] = set()
        for assessment in match.assessments:
            for ref in assessment.evidence_refs:
                if ref in checked:
                    continue
                checked.add(ref)
                if not store.resource_exists(report.patient_id, ref):
                    issues.append(
                        GroundingIssue(
                            kind="unknown_evidence_ref",
                            nct_id=nct,
                            detail=f"Evidence ref {ref!r} is not in the patient's record "
                            f"(criterion: {assessment.criterion[:120]!r}).",
                        )
                    )

        for assessment in match.assessments:
            for found in (_lab_check_issue(assessment, nct, record),
                          _medication_issue(assessment, nct, record)):
                if found is not None:
                    issues.append(found)

        hard = hard_results.get(nct)
        if hard is not None and not hard.passed and match.verdict != "likely_ineligible":
            failed = sorted(k for k, v in hard.checks.items() if v == "fail")
            issues.append(
                GroundingIssue(
                    kind="hard_check_contradiction",
                    nct_id=nct,
                    detail=f"Verdict is {match.verdict} but code hard checks failed: "
                    f"{', '.join(failed)}. {' '.join(hard.notes)}".strip(),
                )
            )

    return issues
