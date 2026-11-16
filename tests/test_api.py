"""HTTP API tests with an injected fake run function: no agent, no key, no network."""

from __future__ import annotations

import dataclasses
import threading
import time

import pytest
from fastapi.testclient import TestClient

from trialmatch.api import JobStore, create_app
from trialmatch.config import Settings
from trialmatch.models import (
    CriterionAssessment,
    MatchReport,
    MatchResult,
    RunTrace,
    TrialMatch,
    Usage,
)


def canned_result(patient_id: str, status: str = "completed") -> MatchResult:
    report = MatchReport(
        patient_id=patient_id,
        trials_considered=1,
        matches=[
            TrialMatch(
                nct_id="NCT00000001",
                title="A study of drug X in adults with type 2 diabetes",
                verdict="likely_eligible",
                rank=1,
                summary="Meets age, diagnosis and HbA1c range.",
                assessments=[
                    CriterionAssessment(
                        criterion="HbA1c between 7.0% and 10.5%",
                        kind="inclusion",
                        status="met",
                        evidence_refs=["Observation/o-hba1c-2"],
                        rationale="HbA1c 8.1%.",
                    )
                ],
            )
        ],
    )
    trace = RunTrace(
        run_id="agent-run-1",
        patient_id=patient_id,
        model="claude-opus-5-5",
        turns=3,
        usage=Usage(input_tokens=300, output_tokens=150),
        wall_seconds=1.5,
    )
    return MatchResult(status=status, report=report, trace=trace)


class FakeRun:
    def __init__(self, status: str = "completed", raises: Exception | None = None):
        self.status = status
        self.raises = raises
        self.calls: list[tuple[str, dict]] = []
        self.release = threading.Event()
        self.release.set()

    def __call__(self, patient_id: str, **kwargs) -> MatchResult:
        self.calls.append((patient_id, kwargs))
        self.release.wait(5)
        if self.raises is not None:
            raise self.raises
        return canned_result(patient_id, self.status)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(patient_dir=tmp_path, http_mode="replay", max_concurrent_runs=1)


def make_client(fake_store, fake_trials, settings, run_fn) -> TestClient:
    app = create_app(run_fn=run_fn, store=fake_store, trials=fake_trials, settings=settings)
    return TestClient(app)


def wait_done(client: TestClient, run_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/matches/{run_id}").json()
        if body["status"] not in ("queued", "running"):
            return body
        time.sleep(0.01)
    raise AssertionError(f"run {run_id} did not finish")


def test_job_lifecycle(fake_store, fake_trials, settings):
    run = FakeRun()
    run.release.clear()
    with make_client(fake_store, fake_trials, settings, run) as client:
        r = client.post("/matches", json={"patient_id": "test-001", "country": "Netherlands"})
        assert r.status_code == 202
        body = r.json()
        assert body["status"] == "queued"
        run_id = body["run_id"]

        pending = client.get(f"/matches/{run_id}").json()
        assert pending["status"] in ("queued", "running")
        assert pending["result"] is None

        run.release.set()
        done = wait_done(client, run_id)
    assert done["status"] == "completed"
    assert done["result"]["report"]["matches"][0]["nct_id"] == "NCT00000001"
    patient_id, kwargs = run.calls[0]
    assert patient_id == "test-001"
    assert kwargs["country"] == "Netherlands"
    assert kwargs["effort"] == "high"  # settings default
    assert kwargs["store"] is fake_store and kwargs["trials"] is fake_trials
    assert kwargs["model"] == settings.model and kwargs["max_turns"] == settings.max_turns


@pytest.mark.parametrize("status", ["completed_with_issues", "refused", "incomplete"])
def test_non_completed_status_is_surfaced(fake_store, fake_trials, settings, status):
    with make_client(fake_store, fake_trials, settings, FakeRun(status=status)) as client:
        run_id = client.post("/matches", json={"patient_id": "test-001"}).json()["run_id"]
        assert wait_done(client, run_id)["status"] == status


def test_raising_run_becomes_error_and_worker_survives(fake_store, fake_trials, settings):
    boom = FakeRun(raises=RuntimeError("kaboom"))
    with make_client(fake_store, fake_trials, settings, boom) as client:
        run_id = client.post("/matches", json={"patient_id": "test-001"}).json()["run_id"]
        body = wait_done(client, run_id)
        assert body["status"] == "error"
        assert "RuntimeError" in body["result"]["error"]
        assert "kaboom" not in body["result"]["error"]  # messages may quote patient data

        # The single worker thread is still alive and takes the next job.
        boom.raises = None
        run_id2 = client.post("/matches", json={"patient_id": "test-001"}).json()["run_id"]
        assert wait_done(client, run_id2)["status"] == "completed"


def test_effort_override_and_validation(fake_store, fake_trials, settings):
    run = FakeRun()
    with make_client(fake_store, fake_trials, settings, run) as client:
        ok = client.post("/matches", json={"patient_id": "test-001", "effort": "xhigh"})
        assert ok.status_code == 202
        wait_done(client, ok.json()["run_id"])
        assert run.calls[0][1]["effort"] == "xhigh"

        bad = client.post("/matches", json={"patient_id": "test-001", "effort": "extreme"})
        assert bad.status_code == 422
    assert len(run.calls) == 1


def test_unknown_patient_is_404_and_not_queued(fake_store, fake_trials, settings):
    run = FakeRun()
    with make_client(fake_store, fake_trials, settings, run) as client:
        r = client.post("/matches", json={"patient_id": "nobody"})
        assert r.status_code == 404
        assert client.post("/matches", json={}).status_code == 422
    assert run.calls == []


def test_unknown_run_is_404(fake_store, fake_trials, settings):
    with make_client(fake_store, fake_trials, settings, FakeRun()) as client:
        assert client.get("/matches/does-not-exist").status_code == 404


def test_patients_listing(fake_store, fake_trials, settings):
    with make_client(fake_store, fake_trials, settings, FakeRun()) as client:
        body = client.get("/patients").json()
    assert body == [
        {
            "patient_id": "test-001",
            "age_years": 56,
            "sex": "female",
            "primary_condition": "Type 2 diabetes mellitus",
            "summary": "female, 56y, Type 2 diabetes mellitus",
        }
    ]


def test_patients_from_bundled_data(monkeypatch, fake_trials):
    # Default settings and a lazily built FhirPatientStore over data/patients.
    monkeypatch.delenv("TRIALMATCH_PATIENT_DIR", raising=False)
    with TestClient(create_app(run_fn=FakeRun(), trials=fake_trials)) as client:
        ids = [p["patient_id"] for p in client.get("/patients").json()]
    assert "p001" in ids


def test_missing_patient_dir_is_503(tmp_path, fake_trials):
    settings = Settings(patient_dir=tmp_path / "missing")
    with TestClient(create_app(run_fn=FakeRun(), trials=fake_trials, settings=settings)) as c:
        assert c.get("/patients").status_code == 503
        assert c.get("/healthz").status_code == 200


def test_healthz_reports_presence_never_the_key(monkeypatch, fake_store, fake_trials, settings):
    secret = "sk-ant-test-SECRET-value-123"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    with make_client(fake_store, fake_trials, settings, FakeRun()) as client:
        r = client.get("/healthz")
    assert r.status_code == 200
    assert secret not in r.text
    assert r.json() == {
        "status": "ok",
        "model": "claude-opus-5-5",
        "http_mode": "replay",
        "api_key_configured": True,
    }


@pytest.fixture
def no_credentials(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr("trialmatch.config._auto_discoverable_credentials", lambda: False)


def test_healthz_without_key(no_credentials, fake_store, fake_trials, settings):
    with make_client(fake_store, fake_trials, settings, FakeRun()) as client:
        assert client.get("/healthz").json()["api_key_configured"] is False


def test_logs_have_metrics_but_no_patient_data(caplog, fake_store, fake_trials, settings):
    caplog.set_level("DEBUG", logger="trialmatch")
    leaky = ValueError("HbA1c 9.7% out of range for patient with Type 2 diabetes mellitus")
    run = FakeRun(raises=leaky)
    with make_client(fake_store, fake_trials, settings, run) as client:
        crashed = client.post("/matches", json={"patient_id": "test-001"}).json()["run_id"]
        body = wait_done(client, crashed)
        run.raises = None
        run_id = client.post("/matches", json={"patient_id": "test-001"}).json()["run_id"]
        wait_done(client, run_id)
    assert body["status"] == "error"
    assert "ValueError" in body["result"]["error"] and "9.7" not in body["result"]["error"]
    text = caplog.text
    assert '"event": "run_crashed"' in text and '"error_type": "ValueError"' in text
    assert '"event": "run_finished"' in text
    assert run_id in text and '"turns": 3' in text and '"wall_seconds"' in text
    for leaked in ("diabetes", "Metformin", "1970-05-01", "HbA1c", "9.7", "Traceback"):
        assert leaked not in text
    assert all(r.exc_info is None for r in caplog.records)


def test_real_agent_without_credentials_is_503(no_credentials, fake_store, fake_trials, settings):
    # No run_fn: the real agent would run, so refuse up front instead of queueing a doomed run.
    app = create_app(store=fake_store, trials=fake_trials, settings=settings)
    with TestClient(app) as client:
        r = client.post("/matches", json={"patient_id": "test-001"})
    assert r.status_code == 503
    assert "ANTHROPIC_API_KEY" in r.json()["detail"]


def test_module_app_imports_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import trialmatch.api as api

    with TestClient(api.app) as client:
        assert client.get("/healthz").status_code == 200


def test_job_store_evicts_oldest_finished():
    store = JobStore(max_finished=2)
    ids = []
    for _ in range(3):
        job = store.create("p")
        store.update(job.run_id, "completed")
        ids.append(job.run_id)
    store.create("p")  # triggers eviction
    assert store.get(ids[0]) is None
    assert store.get(ids[2]) is not None


# --------------------------------------------------------------------------- auth


TOKEN = "test-token-abc123"


@pytest.fixture
def token_settings(settings) -> Settings:
    return dataclasses.replace(settings, api_token=TOKEN)


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("get", "/patients", None),
        ("post", "/matches", {"patient_id": "test-001"}),
        ("get", "/matches/some-run", None),
    ],
)
@pytest.mark.parametrize("header", [None, "Bearer wrong-token", TOKEN, "Basic " + TOKEN])
def test_auth_rejects_missing_or_wrong_token(
    fake_store, fake_trials, token_settings, method, path, body, header
):
    run = FakeRun()
    headers = {"Authorization": header} if header else {}
    with make_client(fake_store, fake_trials, token_settings, run) as client:
        r = client.request(method, path, json=body, headers=headers)
    assert r.status_code == 401
    assert r.headers["WWW-Authenticate"] == "Bearer"
    assert run.calls == []


def test_auth_accepts_right_token_and_healthz_stays_open(fake_store, fake_trials, token_settings):
    auth = {"Authorization": f"Bearer {TOKEN}"}
    with make_client(fake_store, fake_trials, token_settings, FakeRun()) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/patients", headers=auth).status_code == 200
        r = client.post("/matches", json={"patient_id": "test-001"}, headers=auth)
        assert r.status_code == 202
        run_id = r.json()["run_id"]
        assert client.get(f"/matches/{run_id}", headers=auth).status_code == 200


def test_settings_read_token_and_never_print_it():
    env = {"TRIALMATCH_API_TOKEN": "  sekret-value  ", "TRIALMATCH_MAX_QUEUED_RUNS": "4"}
    s = Settings.from_env(env)
    assert s.api_token == "sekret-value" and s.max_queued_runs == 4
    assert "sekret-value" not in repr(s)
    assert Settings.from_env({}).api_token is None


def test_unauthenticated_startup_warns_once(
    caplog, fake_store, fake_trials, settings, token_settings
):
    caplog.set_level("WARNING", logger="trialmatch")
    with make_client(fake_store, fake_trials, settings, FakeRun()):
        pass
    warnings = [r for r in caplog.records if "unauthenticated" in r.getMessage()]
    assert len(warnings) == 1
    caplog.clear()
    with make_client(fake_store, fake_trials, token_settings, FakeRun()):
        pass
    assert not [r for r in caplog.records if "unauthenticated" in r.getMessage()]


# --------------------------------------------------------------------------- queue cap


def test_full_queue_is_429(fake_store, fake_trials, settings):
    capped = dataclasses.replace(settings, max_concurrent_runs=1, max_queued_runs=2)
    run = FakeRun()
    run.release.clear()  # block the worker: jobs stay running or queued
    with make_client(fake_store, fake_trials, capped, run) as client:
        ids = [
            client.post("/matches", json={"patient_id": "test-001"}).json()["run_id"]
            for _ in range(2)
        ]
        full = client.post("/matches", json={"patient_id": "test-001"})
        assert full.status_code == 429
        assert "Retry-After" in full.headers

        run.release.set()
        for run_id in ids:
            assert wait_done(client, run_id)["status"] == "completed"
        # Capacity frees up once jobs finish.
        assert client.post("/matches", json={"patient_id": "test-001"}).status_code == 202


def test_job_store_caps_active_jobs():
    store = JobStore()
    assert store.create("p", max_active=1) is not None
    assert store.create("p", max_active=1) is None
