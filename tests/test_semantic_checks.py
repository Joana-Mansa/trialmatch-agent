"""Checks on what a citation says: lab thresholds recomputed, drug classes matched.

Uses the real synthetic records: p002 (eGFR 24 mL/min/1.73m2) and p004 (tamoxifen),
including the exact mistake qwen3:4b made on p004.
"""

from __future__ import annotations

import pytest

from trialmatch import drug_classes as dc
from trialmatch.grounding import validate
from trialmatch.models import CriterionAssessment, LabCheck, MatchReport, TrialMatch
from trialmatch.patients import FhirPatientStore

NCT = "NCT00000001"
EGFR = "Observation/p002-obs-7"  # latest eGFR, 24.0
TAMOXIFEN = "MedicationRequest/p004-med-1"


@pytest.fixture(scope="module")
def store():
    return FhirPatientStore()


def _issues(store, patient, assessment, verdict="possibly_eligible"):
    report = MatchReport(
        patient_id=patient,
        matches=[TrialMatch(nct_id=NCT, title="t", verdict=verdict, rank=1, summary="s",
                            assessments=[assessment])],
        trials_considered=1,
    )
    return [i for i in validate(report, store, {NCT}, {})
            if i.kind in ("lab_value_contradiction", "lab_check_invalid", "evidence_mismatch")]


def _egfr(status, op=">=", value=30.0, unit="mL/min/1.73m2", kind="inclusion", refs=(EGFR,),
          lab_ref=EGFR):
    return CriterionAssessment(
        criterion="eGFR >= 30 mL/min/1.73m2", kind=kind, status=status,
        evidence_refs=list(refs), rationale="r",
        lab_check=LabCheck(lab_ref=lab_ref, operator=op, value=value, unit=unit),
    )


# ------------------------------------------------------------------ lab thresholds


def test_lab_claim_that_matches_the_value_passes(store):
    assert _issues(store, "p002", _egfr("not_met"), verdict="likely_ineligible") == []


def test_lab_claim_that_contradicts_the_value_is_flagged(store):
    issues = _issues(store, "p002", _egfr("met"))
    assert [i.kind for i in issues] == ["lab_value_contradiction"]
    assert "24 >= 30 is False" in issues[0].detail and "not_met" in issues[0].detail


def test_exclusion_condition_is_read_as_written(store):
    # Exclusion "eGFR < 30": the condition holds for 24, so the exclusion is met.
    assert _issues(store, "p002", _egfr("met", op="<", kind="exclusion"),
                   verdict="likely_ineligible") == []
    assert [i.kind for i in _issues(store, "p002", _egfr("not_met", op="<",
                                                         kind="exclusion"))] == [
        "lab_value_contradiction"]


def test_unknown_is_allowed(store):
    assert _issues(store, "p002", _egfr("unknown")) == []


def test_unit_mismatch_and_bad_refs_are_invalid(store):
    assert [i.kind for i in _issues(store, "p002", _egfr("not_met", unit="mL/s"))] == [
        "lab_check_invalid"]
    # Spacing and case differences in units are not a mismatch.
    assert _issues(store, "p002", _egfr("not_met", unit="ml/min/1.73 m2"),
                   verdict="likely_ineligible") == []
    not_cited = _egfr("not_met", refs=("Condition/p002-cond-1",))
    assert "must also be in evidence_refs" in _issues(store, "p002", not_cited)[0].detail
    not_a_lab = _egfr("not_met", refs=(TAMOXIFEN,), lab_ref=TAMOXIFEN)
    assert "not a lab result" in _issues(store, "p004", not_a_lab)[0].detail


def test_older_values_in_the_history_can_be_cited(store):
    # p002-obs-4 is eGFR 29 from earlier; still a real lab, still recomputed.
    old = _egfr("met", refs=("Observation/p002-obs-4",), lab_ref="Observation/p002-obs-4")
    assert [i.kind for i in _issues(store, "p002", old)] == ["lab_value_contradiction"]


# ------------------------------------------------------------------ drug classes


def _med(criterion, status="met", refs=(TAMOXIFEN,), kind="inclusion"):
    return CriterionAssessment(criterion=criterion, kind=kind, status=status,
                               evidence_refs=list(refs), rationale="r")


@pytest.mark.parametrize("criterion", [
    # Verbatim from the qwen3:4b report in examples/p004-qwen3-4b.json.
    "Must have received ≤5 prior lines of treatment for advanced disease including "
    "cyclin-dependent kinase 4/6 inhibitor",
    "Prior treatment must include a CDK 4/6 inhibitor",
])
def test_qwen_tamoxifen_as_cdk46_inhibitor_is_flagged(store, criterion):
    issues = _issues(store, "p004", _med(criterion))
    assert [i.kind for i in issues] == ["evidence_mismatch"]
    assert "tamoxifen" in issues[0].detail and "CDK4/6 inhibitor" in issues[0].detail


def test_saved_local_model_report_is_now_flagged(store):
    import json
    from pathlib import Path

    from trialmatch.models import MatchResult

    path = Path(__file__).parents[1] / "examples" / "p004-qwen3-4b.json"
    report = MatchResult.model_validate(json.loads(path.read_text("utf-8"))).report
    issues = validate(report, store, {m.nct_id for m in report.matches}, {})
    assert [(i.kind, i.nct_id) for i in issues if i.kind == "evidence_mismatch"] == [
        ("evidence_mismatch", "NCT07213791")]


@pytest.mark.parametrize("criterion", [
    "Prior endocrine therapy for advanced disease",
    "Prior tamoxifen allowed",
    "Received at least one line of hormonal therapy",
    "Prior SERMs or aromatase inhibitors",
])
def test_medication_in_the_named_class_passes(store, criterion):
    assert _issues(store, "p004", _med(criterion)) == []


def test_only_met_claims_and_known_drugs_are_judged(store):
    assert _issues(store, "p004", _med("Prior CDK4/6 inhibitor", status="not_met")) == []
    assert _issues(store, "p004", _med("Adequate organ function")) == []  # no class named


def test_unknown_medication_is_never_flagged():
    assert dc.unsupported_medication_claim("Prior CDK4/6 inhibitor",
                                           ["Investigational compound XY-12"]) is None


def test_plural_and_case_insensitive_matching():
    assert [c.name for c in dc.classes_named("No prior Aromatase Inhibitors")] == [
        "aromatase inhibitor"]
    assert dc.drugs_in("Insulin glargine 100 UNT/ML") == {"insulin"}
    assert dc.drugs_in("Metformin hydrochloride") == {"metformin"}


def test_tool_schema_carries_lab_check():
    from trialmatch.tools import TOOLS

    submit = next(t for t in TOOLS if t["name"] == "submit_match_report")
    item = submit["input_schema"]["properties"]["matches"]["items"]
    assessment = item["properties"]["assessments"]["items"]
    assert "lab_check" in assessment["required"]
    obj = assessment["properties"]["lab_check"]["anyOf"][0]
    assert obj["additionalProperties"] is False and set(obj["required"]) == {
        "lab_ref", "operator", "value", "unit"}
