"""CLI tests: the run function and stores are injected, so no agent, key or network."""

from __future__ import annotations

import io
import json

import pytest
from test_api import canned_result

from trialmatch import cli
from trialmatch.models import GroundingIssue, MatchResult


def run_cli(argv, **kwargs) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(argv, stdout=out, stderr=err, **kwargs)
    return code, out.getvalue(), err.getvalue()


def fake_run(status: str = "completed", calls: list | None = None):
    def run(patient_id: str, **kwargs) -> MatchResult:
        if calls is not None:
            calls.append((patient_id, kwargs))
        result = canned_result(patient_id, status)
        result.grounding_issues = [
            GroundingIssue(kind="unknown_trial", nct_id="NCT99999999", detail="never retrieved")
        ]
        result.report.matches[0].missing_info = ["Recent pregnancy test"]
        return result

    return run


@pytest.fixture
def no_credentials(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr("trialmatch.config._auto_discoverable_credentials", lambda: False)


def test_patients_with_injected_store(fake_store):
    code, out, _ = run_cli(["patients"], store=fake_store)
    assert code == 0
    assert out.split() == ["test-001", "female", "56y", "Type", "2", "diabetes", "mellitus"]


def test_patients_from_bundled_data(monkeypatch):
    monkeypatch.delenv("TRIALMATCH_PATIENT_DIR", raising=False)
    code, out, _ = run_cli(["patients"])
    assert code == 0
    lines = out.strip().splitlines()
    assert lines[0].startswith("p001")
    assert len(lines) >= 8


def test_match_json_and_out_file(fake_store, fake_trials, tmp_path):
    calls: list = []
    target = tmp_path / "result.json"
    code, out, _ = run_cli(
        [
            "match",
            "--patient",
            "test-001",
            "--country",
            "Netherlands",
            "--effort",
            "low",
            "--json",
            "--out",
            str(target),
        ],
        run_fn=fake_run(calls=calls),
        store=fake_store,
        trials=fake_trials,
    )
    assert code == 0
    parsed = MatchResult.model_validate(json.loads(out))
    assert parsed.status == "completed"
    assert parsed.report.matches[0].nct_id == "NCT00000001"
    assert MatchResult.model_validate_json(target.read_text(encoding="utf-8")) == parsed
    patient_id, kwargs = calls[0]
    assert patient_id == "test-001"
    assert kwargs["country"] == "Netherlands" and kwargs["effort"] == "low"
    assert kwargs["store"] is fake_store and kwargs["trials"] is fake_trials


def test_match_readable_report(fake_store, fake_trials):
    code, out, _ = run_cli(
        ["match", "--patient", "test-001"],
        run_fn=fake_run(),
        store=fake_store,
        trials=fake_trials,
    )
    assert code == 0
    assert "#1  NCT00000001  likely_eligible" in out
    assert "Meets age, diagnosis and HbA1c range." in out
    assert "[met] (inclusion) HbA1c between 7.0% and 10.5%  <- Observation/o-hba1c-2" in out
    assert "Missing info:" in out and "Recent pregnancy test" in out
    assert "Grounding issues:" in out and "unknown_trial NCT99999999" in out
    assert "turns 3 | tool calls 0 | tokens in 300 out 150" in out
    assert "wall 1.5s" in out


def test_report_with_issues_has_needs_review_header(fake_store, fake_trials):
    code, out, _ = run_cli(
        ["match", "--patient", "test-001"],
        run_fn=fake_run("completed_with_issues"),
        store=fake_store,
        trials=fake_trials,
    )
    assert code == 3
    head = out.splitlines()[:3]
    assert any(line.startswith("NEEDS REVIEW: grounding issues") for line in head)


def test_clean_report_has_no_needs_review_header():
    assert "NEEDS REVIEW" not in cli.render_report(canned_result("test-001"))


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("completed", 0),
        ("completed_with_issues", 3),
        ("refused", 2),
        ("incomplete", 2),
        ("error", 1),
    ],
)
def test_match_exit_codes(fake_store, fake_trials, status, expected):
    code, _, _ = run_cli(
        ["match", "--patient", "test-001", "--json"],
        run_fn=fake_run(status),
        store=fake_store,
        trials=fake_trials,
    )
    assert code == expected


def test_unknown_patient_exits_1(fake_store, fake_trials):
    calls: list = []
    code, _, err = run_cli(
        ["match", "--patient", "nobody"],
        run_fn=fake_run(calls=calls),
        store=fake_store,
        trials=fake_trials,
    )
    assert code == 1
    assert "Unknown patient id 'nobody'" in err and "test-001" in err
    assert calls == []


def test_missing_key_prints_help_not_traceback(no_credentials, fake_store, fake_trials):
    # No run_fn injected: the real agent path, which must stop before any API call.
    code, out, err = run_cli(
        ["match", "--patient", "test-001"], store=fake_store, trials=fake_trials
    )
    assert code == 1
    assert out == ""
    assert "ANTHROPIC_API_KEY" in err
    assert '$env:ANTHROPIC_API_KEY = "' in err  # PowerShell
    assert 'export ANTHROPIC_API_KEY="' in err  # bash
    assert "Traceback" not in err


def test_bad_effort_is_rejected(fake_store):
    with pytest.raises(SystemExit) as exc:
        run_cli(["match", "--patient", "test-001", "--effort", "extreme"], store=fake_store)
    assert exc.value.code == 2


def test_bad_env_config_is_reported(monkeypatch, fake_store):
    monkeypatch.setenv("TRIALMATCH_MAX_TURNS", "lots")
    code, _, err = run_cli(["patients"], store=fake_store)
    assert code == 1
    assert "TRIALMATCH_MAX_TURNS" in err
