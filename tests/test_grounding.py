from __future__ import annotations

from trialmatch.grounding import validate
from trialmatch.hard_checks import check
from trialmatch.models import CriterionAssessment, MatchReport, TrialMatch

NCT = "NCT00000001"


def _report(verdict="likely_eligible", refs=("Condition/c-t2d",), nct=NCT, assessments=None):
    if assessments is None:
        assessments = [
            CriterionAssessment(
                criterion="Type 2 diabetes",
                kind="inclusion",
                status="met",
                evidence_refs=list(refs),
                rationale="Recorded diagnosis.",
            )
        ]
    return MatchReport(
        patient_id="test-001",
        matches=[
            TrialMatch(
                nct_id=nct,
                title="t",
                verdict=verdict,
                rank=1,
                summary="s",
                assessments=assessments,
            )
        ],
        trials_considered=1,
    )


def _hard(patient, trials, **update):
    elig = trials.eligibility(NCT).model_copy(update=update)
    return {NCT: check(patient, elig)}


def test_clean_report_has_no_issues(fake_store, fake_trials, diabetic_patient):
    refs = ("Condition/c-t2d", "Observation/o-hba1c-1", "Patient/test-001")
    issues = validate(
        _report(refs=refs), fake_store, {NCT}, _hard(diabetic_patient, fake_trials)
    )
    assert issues == []


def test_fabricated_evidence_ref(fake_store, fake_trials, diabetic_patient):
    issues = validate(
        _report(refs=("Condition/c-t2d", "Observation/made-up", "Observation/made-up")),
        fake_store,
        {NCT},
        _hard(diabetic_patient, fake_trials),
    )
    assert [i.kind for i in issues] == ["unknown_evidence_ref"]
    assert "Observation/made-up" in issues[0].detail
    assert issues[0].nct_id == NCT


def test_hard_check_contradiction(fake_store, fake_trials, diabetic_patient):
    hard = _hard(diabetic_patient, fake_trials, maximum_age="50 Years")
    issues = validate(_report(verdict="possibly_eligible"), fake_store, {NCT}, hard)
    assert [i.kind for i in issues] == ["hard_check_contradiction"]
    assert "maximum_age" in issues[0].detail

    assert validate(_report(verdict="likely_ineligible"), fake_store, {NCT}, hard) == []


def test_unknown_trial_and_empty_assessments(fake_store):
    report = _report(nct="NCT12345678", assessments=[], verdict="possibly_eligible")
    issues = validate(report, fake_store, {NCT}, {})
    assert sorted(i.kind for i in issues) == ["empty_assessments", "unknown_trial"]


def _assess(kind, status, refs=("Condition/c-t2d",)):
    return CriterionAssessment(
        criterion=f"{kind} item", kind=kind, status=status, evidence_refs=list(refs),
        rationale="r",
    )


def test_eligible_with_exclusion_met_is_inconsistent(fake_store):
    report = _report(assessments=[_assess("inclusion", "met"), _assess("exclusion", "met")])
    kinds = [i.kind for i in validate(report, fake_store, {NCT}, {})]
    assert kinds == ["verdict_inconsistent"]


def test_possibly_eligible_with_inclusion_not_met_is_inconsistent(fake_store):
    report = _report(verdict="possibly_eligible", assessments=[_assess("inclusion", "not_met")])
    assert [i.kind for i in validate(report, fake_store, {NCT}, {})] == ["verdict_inconsistent"]


def test_eligible_without_evidence_or_met_inclusion(fake_store):
    report = _report(assessments=[_assess("inclusion", "unknown", refs=())])
    kinds = sorted(i.kind for i in validate(report, fake_store, {NCT}, {}))
    assert kinds == ["verdict_inconsistent", "verdict_inconsistent"]


def test_duplicate_trials_and_ranks(fake_store):
    report = _report()
    report.matches.append(report.matches[0].model_copy())
    kinds = sorted(i.kind for i in validate(report, fake_store, {NCT}, {}))
    assert kinds == ["duplicate_trial", "duplicate_trial"]


def test_not_recruiting_and_truncated(fake_store, fake_trials):
    elig = fake_trials.eligibility(NCT).model_copy(
        update={"status": "TERMINATED", "criteria_text": "x\n[truncated: showing 1 of 9]"}
    )
    kinds = sorted(i.kind for i in validate(_report(), fake_store, {NCT}, {}, {NCT: elig}))
    assert kinds == ["criteria_truncated", "not_recruiting"]
