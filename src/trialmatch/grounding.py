"""Code checks on a submitted report: is everything it cites real and consistent?

These checks do not judge clinical reasoning. They catch the mechanical failures a
model can make: citing a resource id that is not in the record, reporting a trial it
never retrieved or that is not recruiting, ignoring a failed hard check, giving a
verdict that contradicts its own criterion assessments, or calling a trial likely
eligible when part of its criteria text was never seen.

The agent runs these checks when the report is submitted and sends any problems back
to the model to fix, then runs them again on the final report.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .models import GroundingIssue, HardCheckResult, MatchReport, TrialEligibility, TrialMatch
from .ports import PatientStore

TRUNCATION_MARKER = "[truncated:"


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
