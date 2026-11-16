from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import FakeTrialsSource
from fakes_anthropic import FakeAnthropic, tool_turn

from trialmatch import eval as ev
from trialmatch.models import (
    CriterionAssessment,
    GroundingIssue,
    MatchReport,
    MatchResult,
    RunTrace,
    TraceStep,
    TrialEligibility,
    TrialHit,
    TrialMatch,
    Usage,
)
from trialmatch.patients import FhirPatientStore

PREG_NCT = "NCT10000001"
RENAL_NCT = "NCT10000002"
ADULT_NCT = "NCT10000003"
PLAIN_NCT = "NCT10000004"

PREG_TEXT = (
    "Inclusion Criteria:\n\n* Women with HR+ breast cancer\n\nExclusion Criteria:\n\n"
    "* Pregnant or breast-feeding women"
)
PLAIN_TEXT = "Inclusion Criteria:\n\n* Breast cancer\n\nExclusion Criteria:\n\n* Active infection"


def _elig(nct: str, text: str, min_age: str | None = "18 Years",
          max_age: str | None = "75 Years") -> TrialEligibility:
    return TrialEligibility(nct_id=nct, title=f"Trial {nct}", criteria_text=text,
                            minimum_age=min_age, maximum_age=max_age)


@pytest.fixture(scope="module")
def store() -> FhirPatientStore:
    return FhirPatientStore()


@pytest.fixture
def trials() -> FakeTrialsSource:
    eligs = {
        PREG_NCT: _elig(PREG_NCT, PREG_TEXT),
        RENAL_NCT: _elig(RENAL_NCT, "Exclusion Criteria:\n\n* eGFR < 30 mL/min/1.73m2"),
        ADULT_NCT: _elig(ADULT_NCT, "Inclusion Criteria:\n\n* Adults", max_age="80 Years"),
        PLAIN_NCT: _elig(PLAIN_NCT, PLAIN_TEXT),
    }
    hits = [TrialHit(nct_id=n, title=e.title, status="RECRUITING") for n, e in eligs.items()]
    return FakeTrialsSource(hits, eligs)


@pytest.fixture(scope="module")
def expectations() -> dict:
    return ev.load_expectations()


def a(kind: str, status: str, refs: list[str], criterion: str = "c") -> CriterionAssessment:
    return CriterionAssessment(criterion=criterion, kind=kind, status=status,
                               evidence_refs=refs, rationale="r")


def match(nct: str, verdict: str, assessments: list[CriterionAssessment]) -> TrialMatch:
    return TrialMatch(nct_id=nct, title=f"Trial {nct}", verdict=verdict, rank=1, summary="s",
                      assessments=assessments)


def result(pid: str, matches: list[TrialMatch] | None, *, status: str = "completed",
           issues: list[GroundingIssue] | None = None, **trace_kw) -> MatchResult:
    report = None if matches is None else MatchReport(patient_id=pid, matches=matches,
                                                      trials_considered=len(matches))
    return MatchResult(status=status, report=report,
                       trace=RunTrace(run_id="r", patient_id=pid, model="claude-opus-5-5",
                                      **trace_kw),
                       grounding_issues=issues or [])


# --------------------------------------------------------------------------- health


def test_run_health_counts_statuses():
    rs = [result("p001", []), result("p002", None, status="error"),
          result("p003", None, status="refused"), result("p004", [])]
    assert ev.run_health(rs) == {
        "completed": 2, "completed_with_issues": 0, "refused": 1, "incomplete": 0, "error": 1,
        "total": 4,
    }


# --------------------------------------------------------------------------- citations


def test_citation_validity_flags_fabricated_ref(store):
    r = result("p004", [match(PREG_NCT, "likely_ineligible", [
        a("exclusion", "met", ["Observation/p004-obs-4", "Observation/made-up-1"]),
        a("inclusion", "met", ["Condition/p004-cond-1"]),
    ])])
    c = ev.citation_validity(r, store)
    assert c["total_refs"] == 3
    assert c["valid_refs"] == 2
    assert c["fabricated_count"] == 1
    assert c["fabricated"] == [{"nct_id": PREG_NCT, "ref": "Observation/made-up-1"}]
    assert c["share_valid"] == pytest.approx(2 / 3, abs=1e-4)


def test_citation_validity_ref_from_other_patient_is_fabricated(store):
    r = result("p001", [match(PLAIN_NCT, "possibly_eligible",
                              [a("inclusion", "met", ["Observation/p004-obs-4"])])])
    assert ev.citation_validity(r, store)["fabricated_count"] == 1


def test_citation_validity_no_refs_is_none(store):
    assert ev.citation_validity(result("p001", None), store)["share_valid"] is None


# --------------------------------------------------------------------------- hard checks


def test_hard_check_contradiction_recomputed(store, trials):
    # p006 is 81; ADULT_NCT has maximum age 80, PREG_NCT 75.
    r = result("p006", [
        match(ADULT_NCT, "likely_eligible", [a("inclusion", "met", ["Patient/p006"])]),
        match(PREG_NCT, "likely_ineligible", [a("inclusion", "not_met", ["Patient/p006"])]),
    ])
    h = ev.hard_constraint_violations(r, store, trials)
    assert h["checked"] == 2
    assert h["count"] == 1
    v = h["violations"][0]
    assert v["nct_id"] == ADULT_NCT
    assert v["failed_checks"] == ["maximum_age"]
    assert v["sources"] == ["recomputed"]


def test_hard_check_merges_grounding_issue_and_handles_missing_trial(store, trials):
    issue = GroundingIssue(kind="hard_check_contradiction", nct_id=ADULT_NCT, detail="x")
    r = result("p006", [
        match(ADULT_NCT, "possibly_eligible", [a("inclusion", "met", [])]),
        match("NCT99999999", "possibly_eligible", [a("inclusion", "met", [])]),
    ], issues=[issue])
    h = ev.hard_constraint_violations(r, store, trials)
    assert h["count"] == 1
    assert h["violations"][0]["sources"] == ["grounding", "recomputed"]
    assert h["unverified"] == ["NCT99999999"]


def test_hard_check_clean_patient_has_no_violations(store, trials):
    r = result("p001", [match(ADULT_NCT, "likely_eligible", [a("inclusion", "met", [])])])
    assert ev.hard_constraint_violations(r, store, trials)["count"] == 0


# --------------------------------------------------------------------------- coverage


def test_criterion_coverage():
    r = result("p001", [
        match(PLAIN_NCT, "possibly_eligible", [
            a("inclusion", "met", ["Condition/x"]),
            a("inclusion", "unknown", []),
            a("exclusion", "not_met", ["Observation/y"]),
            a("exclusion", "unknown", []),
        ]),
        match(ADULT_NCT, "likely_eligible", [a("inclusion", "met", ["Patient/p001"])]),
    ])
    c = ev.criterion_coverage(r)
    assert c["matches"] == 2
    assert c["per_match"][0] == {"nct_id": PLAIN_NCT, "verdict": "possibly_eligible",
                                 "assessments": 4, "unknown_share": 0.5,
                                 "evidence_share": 0.5}
    assert c["mean_assessments"] == 2.5
    assert c["mean_unknown_share"] == 0.25
    assert c["mean_evidence_share"] == 0.75


# --------------------------------------------------------------------------- planted recall


def test_planted_pregnancy_found(store, trials, expectations):
    r = result("p004", [match(PREG_NCT, "likely_ineligible", [
        a("exclusion", "met", ["Observation/p004-obs-4"], "Pregnant or breast-feeding"),
    ])])
    p = ev.planted_recall(r, store, trials, expectations)
    assert (p["relevant"], p["hits"], p["recall"]) == (1, 1, 1.0)
    assert p["trials"][0]["hit"] is True


def test_planted_pregnancy_missed(store, trials, expectations):
    r = result("p004", [match(PREG_NCT, "likely_eligible", [
        a("exclusion", "not_met", ["Observation/p004-obs-4"], "Pregnant or breast-feeding"),
    ])])
    p = ev.planted_recall(r, store, trials, expectations)
    assert (p["relevant"], p["hits"], p["recall"]) == (1, 0, 0.0)


def test_planted_pregnancy_flagged_with_wrong_ref_or_eligible_verdict_is_miss(
    store, trials, expectations
):
    wrong_ref = match(PREG_NCT, "likely_ineligible",
                      [a("exclusion", "met", ["Condition/p004-cond-1"])])
    eligible = match(PREG_NCT, "likely_eligible",
                     [a("exclusion", "met", ["Condition/p004-cond-2"])])
    for m in (wrong_ref, eligible):
        p = ev.planted_recall(result("p004", [m]), store, trials, expectations)
        assert p["hits"] == 0 and p["relevant"] == 1


def test_planted_na_when_no_trial_mentions_issue(store, trials, expectations):
    r = result("p004", [match(PLAIN_NCT, "possibly_eligible", [a("inclusion", "met", [])])])
    p = ev.planted_recall(r, store, trials, expectations)
    assert p["relevant"] == 0
    assert p["recall"] is None
    assert "no assessed trial" in p["reason"]


def test_planted_renal_inclusion_not_met_counts(store, trials, expectations):
    r = result("p002", [match(RENAL_NCT, "likely_ineligible", [
        a("inclusion", "not_met", ["Observation/p002-obs-7"], "eGFR >= 30"),
    ])])
    assert ev.planted_recall(r, store, trials, expectations)["recall"] == 1.0


def test_planted_flag_with_lenient_verdict_is_a_miss(store, trials, expectations):
    # A blocking item requires likely_ineligible; possibly_eligible is not a catch.
    r = result("p002", [match(RENAL_NCT, "possibly_eligible", [
        a("inclusion", "not_met", ["Observation/p002-obs-7"], "eGFR >= 30"),
    ])])
    assert ev.planted_recall(r, store, trials, expectations)["recall"] == 0.0


def test_planted_age_edge_uses_hard_check(store, trials, expectations):
    # p007 is 17: every trial here has minimum age 18, so all count.
    r = result("p007", [
        match(ADULT_NCT, "likely_ineligible", [a("inclusion", "not_met", ["Patient/p007"])]),
        match(PLAIN_NCT, "possibly_eligible", [a("inclusion", "unknown", [])]),
    ])
    p = ev.planted_recall(r, store, trials, expectations)
    assert (p["relevant"], p["hits"], p["recall"]) == (2, 1, 0.5)


def test_planted_none_for_clean_patient_and_no_report(store, trials, expectations):
    assert ev.planted_recall(result("p001", []), store, trials, expectations) is None
    p = ev.planted_recall(result("p004", None, status="error"), store, trials, expectations)
    assert p["recall"] is None and p["reason"] == "no report"


def test_expectations_refs_exist_in_store(store, expectations):
    for pid, rule in expectations.items():
        for ref in rule.get("correct_refs", []):
            assert store.resource_exists(pid, ref), (pid, ref)


# --------------------------------------------------------------------------- cost


def test_usage_cost_arithmetic():
    u = Usage(input_tokens=1_000_000, output_tokens=100_000, cache_read_input_tokens=2_000_000,
              cache_creation_input_tokens=400_000)
    # 4.00 + 2.00 + 0.40 + 0.4 * 5.00
    assert ev.usage_cost(u, "claude-opus-5-5") == pytest.approx(8.40)
    assert ev.usage_cost(u, "some-other-model") is None
    assert ev.PRICING["taken"] == "2026-11"
    assert ev.PRICING["per_mtok"]["claude-opus-5-5"]["cache_write"] == 5.0


def test_cost_latency_counts_tools_and_fallback():
    steps = [TraceStep(step=i, tool=t, input={}, ok=t != "search_trials", duration_ms=1.0)
             for i, t in enumerate(["get_patient_summary", "search_trials", "search_trials"])]
    r = result("p001", None, turns=3, steps=steps, fallback_used=True, wall_seconds=12.5,
               usage=Usage(input_tokens=1000, output_tokens=500))
    c = ev.cost_latency(r)
    assert c["tool_calls"] == {"get_patient_summary": 1, "search_trials": 2}
    assert c["tool_errors"] == 2
    assert c["fallback_used"] is True
    assert c["turns"] == 3 and c["wall_seconds"] == 12.5
    assert c["cost_usd"] == pytest.approx(0.014)


# --------------------------------------------------------------------------- gold


def test_gold_loader_skips_comments_and_placeholders(tmp_path):
    p = tmp_path / "gold.jsonl"
    p.write_text(
        "# header\n\n"
        '{"patient_id": "p001", "nct_id": "NCT1", "expected_verdict": "likely_eligible", '
        '"placeholder": true}\n'
        '{"patient_id": "p004", "nct_id": "NCT10000001", '
        '"expected_verdict": "likely_ineligible", "notes": "pregnant"}\n',
        encoding="utf-8",
    )
    rows = ev.load_gold(p)
    assert [r["patient_id"] for r in rows] == ["p004"]
    assert ev.load_gold(tmp_path / "missing.jsonl") == []


def test_shipped_gold_has_no_active_rows():
    assert ev.load_gold() == []


def test_gold_agreement():
    rs = [result("p004", [match(PREG_NCT, "likely_ineligible", []),
                          match(PLAIN_NCT, "likely_eligible", [])])]
    gold = [
        {"patient_id": "p004", "nct_id": PREG_NCT, "expected_verdict": "likely_ineligible"},
        {"patient_id": "p004", "nct_id": PLAIN_NCT, "expected_verdict": "possibly_eligible"},
        {"patient_id": "p004", "nct_id": "NCT55555555", "expected_verdict": "likely_eligible"},
    ]
    g = ev.gold_agreement(rs, gold)
    assert (g["gold_rows"], g["compared"], g["agreed"], g["agreement"]) == (3, 2, 1, 0.5)


# --------------------------------------------------------------------------- summary


def test_summarize_and_render(store, trials, expectations):
    rs = [
        result("p004", [match(PREG_NCT, "likely_ineligible", [
            a("exclusion", "met", ["Observation/p004-obs-4", "Observation/fake-9"]),
        ])], usage=Usage(input_tokens=1000)),
        result("p006", [match(ADULT_NCT, "likely_eligible", [a("inclusion", "met", [])])]),
        result("p001", None, status="error"),
    ]
    s = ev.summarize(rs, store, trials, expectations)
    o = s["overall"]
    assert o["health"]["completed"] == 2 and o["health"]["error"] == 1
    assert o["hard_constraint_violations"] == 1
    assert o["citations"] == {"total_refs": 2, "valid_refs": 1, "fabricated_count": 1,
                              "share_valid": 0.5}
    # p004 hit 1/1, p006 hard check failed and verdict eligible: 0/1
    assert o["planted_recall"] == {"relevant": 2, "hits": 1, "recall": 0.5}
    assert o["cost"]["total_usd"] == pytest.approx(0.004)
    json.dumps(s, default=str)

    md = ev.render_markdown(s, {"model": "claude-opus-5-5"})
    assert "Observation/fake-9" in md
    assert "Hard-constraint violations" in md
    assert chr(0x2014) not in md


# --------------------------------------------------------------------------- end to end


def _p004_script() -> list:
    report = {
        "matches": [{
            "nct_id": PREG_NCT, "title": f"Trial {PREG_NCT}", "verdict": "likely_ineligible",
            "rank": 1, "summary": "Patient is pregnant.",
            "assessments": [
                {"criterion": "Women with HR+ breast cancer", "kind": "inclusion",
                 "status": "met", "evidence_refs": ["Condition/p004-cond-1"],
                 "rationale": "Active HR+ breast cancer."},
                {"criterion": "Pregnant or breast-feeding women", "kind": "exclusion",
                 "status": "met", "evidence_refs": ["Observation/p004-obs-4"],
                 "rationale": "Pregnancy status observation says pregnant."},
            ],
            "missing_info": [],
        }],
        "trials_considered": 1,
        "notes": "",
    }
    return [
        tool_turn(("get_patient_summary", {})),
        tool_turn(("search_trials", {"condition": "breast cancer", "country": None,
                                     "page_size": None})),
        tool_turn(("get_trial_eligibility", {"nct_id": PREG_NCT})),
        tool_turn(("submit_match_report", report)),
    ]


def test_main_end_to_end(tmp_path, store, trials, capsys):
    client = FakeAnthropic(_p004_script())
    rc = ev.main(["--patients", "p004", "--yes", "--out", str(tmp_path)],
                 store=store, trials=trials, client=client)
    assert rc == 0
    assert client.beta.messages.remaining == 0

    (run_dir,) = list(tmp_path.iterdir())
    data = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    o = data["overall"]
    assert o["health"]["completed"] == 1
    assert o["hard_constraint_violations"] == 0
    assert o["citations"]["share_valid"] == 1.0
    assert o["planted_recall"]["recall"] == 1.0
    assert o["cost"]["tool_calls"]["submit_match_report"] == 1
    # 4 turns x (100 in, 50 out, 80 cache read, 20 cache write) from the fake usage
    assert o["cost"]["total_usd"] == pytest.approx(
        (400 * 4 + 200 * 20 + 320 * 0.2 + 80 * 5) / 1e6)

    saved = MatchResult.model_validate_json(
        (run_dir / "runs" / "p004.json").read_text(encoding="utf-8"))
    assert saved.report.matches[0].nct_id == PREG_NCT

    md = (run_dir / "report.md").read_text(encoding="utf-8")
    out = capsys.readouterr().out
    assert "About to run 1 patient(s) with model claude-opus-5-5" in out
    assert "# trialmatch-agent eval report" in out
    assert chr(0x2014) not in md


def test_resume_keeps_finished_runs_and_reruns_errors(tmp_path, store, trials, capsys):
    rc = ev.main(["--patients", "p004", "--yes", "--out", str(tmp_path)],
                 store=store, trials=trials, client=FakeAnthropic(_p004_script()))
    assert rc == 0
    (run_dir,) = list(tmp_path.iterdir())

    # A finished run is kept: resuming makes no model calls and rescores it.
    idle = FakeAnthropic([])
    rc = ev.main(["--patients", "p004", "--yes", "--resume", str(run_dir)],
                 store=store, trials=trials, client=idle)
    assert rc == 0 and idle.calls == []
    assert "Keeping 1 finished run(s)" in capsys.readouterr().out
    data = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert data["overall"]["health"]["completed"] == 1

    # An errored run is run again and overwritten.
    errored = MatchResult(status="error", error="killed",
                          trace=RunTrace(run_id="x", patient_id="p004", model="m"))
    (run_dir / "runs" / "p004.json").write_text(errored.model_dump_json(), encoding="utf-8")
    again = FakeAnthropic(_p004_script())
    rc = ev.main(["--patients", "p004", "--yes", "--resume", str(run_dir)],
                 store=store, trials=trials, client=again)
    assert rc == 0 and again.beta.messages.remaining == 0
    saved = MatchResult.model_validate_json(
        (run_dir / "runs" / "p004.json").read_text(encoding="utf-8"))
    assert saved.status == "completed"
    assert len(list(tmp_path.iterdir())) == 1  # no new folder


def test_resume_needs_a_runs_folder(tmp_path, store, trials):
    rc = ev.main(["--patients", "p004", "--yes", "--resume", str(tmp_path / "nope")],
                 store=store, trials=trials, client=FakeAnthropic([]))
    assert rc == 2


def test_main_aborts_without_confirmation(tmp_path, store, trials):
    client = FakeAnthropic([])
    rc = ev.main(["--patients", "p004", "--out", str(tmp_path)], store=store, trials=trials,
                 client=client, input_fn=lambda _: "n")
    assert rc == 1
    assert client.calls == []
    assert list(Path(tmp_path).iterdir()) == []


def test_main_rejects_unknown_patient(tmp_path, store, trials):
    rc = ev.main(["--patients", "p999", "--yes", "--out", str(tmp_path)], store=store,
                 trials=trials, client=FakeAnthropic([]))
    assert rc == 2
