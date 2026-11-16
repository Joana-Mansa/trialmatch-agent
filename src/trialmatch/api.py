"""HTTP service: submit a match run as a job, then poll for its result.

A run takes minutes (many model turns and ClinicalTrials.gov calls), so ``POST
/matches`` only validates the patient and queues the run on a bounded thread pool;
``GET /matches/{run_id}`` reports progress and, when finished, the ``MatchResult``.

Job store: jobs live in an in-memory dict owned by one process. They are lost on
restart and are not shared between uvicorn workers, so run a single worker. A real
deployment would keep jobs in Postgres or Redis and run the agent in a separate
worker pool (e.g. a queue consumer) rather than in the web process.

``create_app`` takes the run function, patient store and trials source as arguments
so tests can inject fakes and run without the agent or an API key. The store and
trials client are otherwise built lazily on first use, so importing this module
(``app`` below, used by ``uvicorn trialmatch.api:app``) needs no key and no data.

Access: if ``TRIALMATCH_API_TOKEN`` is set, every endpoint except ``/healthz``
requires ``Authorization: Bearer <token>``; otherwise the API is open and a warning
is logged at startup. ``POST /matches`` returns 429 once ``TRIALMATCH_MAX_QUEUED_RUNS``
jobs are queued or running, so the in-memory job store stays bounded.

Logging: log lines carry ids, statuses and metrics only. A crashed run logs its
exception type, never the message or traceback, which may quote patient data.
"""

from __future__ import annotations

import hmac
import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from . import config
from .models import MatchResult, PatientSummary, RunTrace
from .ports import PatientNotFound, PatientStore, TrialsSource

logger = logging.getLogger("trialmatch.api")

RunFn = Callable[..., MatchResult]
JobStatus = Literal[
    "queued", "running", "completed", "completed_with_issues", "refused", "incomplete", "error"
]
Effort = Literal["low", "medium", "high", "xhigh", "max"]

MAX_FINISHED_JOBS = 1000  # oldest finished jobs are evicted beyond this


def log_event(event: str, **fields: Any) -> None:
    """One JSON object per line. Callers pass ids and metrics only, never patient data."""
    logger.info(json.dumps({"event": event, **fields}, default=str, sort_keys=True))


def default_run_fn(patient_id: str, **kwargs: Any) -> MatchResult:
    from .agent import run_match  # imported lazily: keeps app import cheap

    return run_match(patient_id, **kwargs)


# --------------------------------------------------------------------------- schemas


class MatchRequest(BaseModel):
    patient_id: str = Field(min_length=1)
    country: str | None = None
    effort: Effort | None = None


class MatchAccepted(BaseModel):
    run_id: str
    status: Literal["queued"] = "queued"


class MatchJob(BaseModel):
    run_id: str
    status: JobStatus
    result: MatchResult | None = None


class PatientListing(BaseModel):
    patient_id: str
    age_years: int
    sex: str
    primary_condition: str | None
    summary: str  # one line, e.g. "female, 56y, Type 2 diabetes mellitus"


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    model: str
    http_mode: str
    api_key_configured: bool


def patient_listing(s: PatientSummary) -> PatientListing:
    active = [c for c in s.conditions if (c.clinical_status or "active") == "active"]
    primary = (active or s.conditions or [None])[0]
    condition = primary.display if primary is not None else None
    return PatientListing(
        patient_id=s.patient_id,
        age_years=s.age_years,
        sex=s.sex,
        primary_condition=condition,
        summary=f"{s.sex}, {s.age_years}y, {condition or 'no recorded conditions'}",
    )


# --------------------------------------------------------------------------- jobs


@dataclass
class Job:
    run_id: str
    patient_id: str
    status: JobStatus = "queued"
    result: MatchResult | None = None
    created: float = field(default_factory=time.time)


class JobStore:
    """Thread-safe, per-process, in-memory. See the module docstring."""

    def __init__(self, max_finished: int = MAX_FINISHED_JOBS):
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._max_finished = max_finished

    def create(self, patient_id: str, max_active: int | None = None) -> Job | None:
        """Add a queued job, or return None if ``max_active`` jobs are already active.

        Finished jobs are evicted beyond ``max_finished``; active (queued or running)
        jobs are bounded by ``max_active``, so the store cannot grow without limit.
        """
        job = Job(run_id=uuid.uuid4().hex, patient_id=patient_id)
        with self._lock:
            if max_active is not None and self._active() >= max_active:
                return None
            self._jobs[job.run_id] = job
            self._evict()
        return job

    def _active(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status in ("queued", "running"))

    def get(self, run_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(run_id)
            return None if job is None else Job(**vars(job))  # snapshot

    def update(self, run_id: str, status: JobStatus, result: MatchResult | None = None) -> None:
        with self._lock:
            job = self._jobs[run_id]
            job.status = status
            if result is not None:
                job.result = result

    def _evict(self) -> None:
        finished = [j.run_id for j in self._jobs.values() if j.status not in ("queued", "running")]
        for run_id in finished[: max(0, len(finished) - self._max_finished)]:
            del self._jobs[run_id]


# --------------------------------------------------------------------------- app


class _Lazy:
    """Build a dependency on first use, once, even under concurrent requests."""

    def __init__(self, value: Any, factory: Callable[[], Any]):
        self._value = value
        self._factory = factory
        self._lock = threading.Lock()

    def get(self) -> Any:
        if self._value is None:
            with self._lock:
                if self._value is None:
                    self._value = self._factory()
        return self._value


def get_store(request: Request) -> PatientStore:
    try:
        return request.app.state.lazy_store.get()
    except (FileNotFoundError, ValueError) as e:
        log_event("store_unavailable", error=type(e).__name__)
        raise HTTPException(
            503, "Patient store unavailable; check TRIALMATCH_PATIENT_DIR."
        ) from None


StoreDep = Annotated[PatientStore, Depends(get_store)]


def require_token(request: Request, authorization: Annotated[str | None, Header()] = None) -> None:
    """Bearer-token check when ``TRIALMATCH_API_TOKEN`` is configured; no-op otherwise."""
    token: str | None = request.app.state.settings.api_token
    if not token:
        return
    scheme, _, supplied = (authorization or "").partition(" ")
    ok = scheme.lower() == "bearer" and hmac.compare_digest(
        supplied.strip().encode("utf-8"), token.encode("utf-8")
    )
    if not ok:
        raise HTTPException(
            401, "Missing or invalid API token.", headers={"WWW-Authenticate": "Bearer"}
        )


Auth = [Depends(require_token)]


def create_app(
    run_fn: RunFn | None = None,
    store: PatientStore | None = None,
    trials: TrialsSource | None = None,
    settings: config.Settings | None = None,
) -> FastAPI:
    settings = settings or config.get_settings()
    config.configure_logging()
    needs_credentials = run_fn is None  # the real agent; injected fakes need no key
    run_fn = run_fn or default_run_fn
    lazy_store = _Lazy(store, lambda: config.make_store(settings))
    lazy_trials = _Lazy(trials, lambda: config.make_trials(settings))
    jobs = JobStore()
    executor = ThreadPoolExecutor(
        max_workers=settings.max_concurrent_runs, thread_name_prefix="trialmatch-run"
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if not settings.api_token:
            logger.warning(
                "TRIALMATCH_API_TOKEN is not set: the API is unauthenticated. "
                "Bind it to localhost only, or set a token."
            )
        yield
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="trialmatch-agent",
        version="0.1.0",
        description="Match synthetic FHIR patients to recruiting ClinicalTrials.gov trials.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.jobs = jobs
    app.state.executor = executor
    app.state.lazy_store = lazy_store

    def execute(run_id: str, patient_id: str, effort: str, country: str | None) -> None:
        jobs.update(run_id, "running")
        log_event("run_started", run_id=run_id, patient_id=patient_id, effort=effort)
        started = time.perf_counter()
        try:
            result = run_fn(
                patient_id,
                store=lazy_store.get(),
                trials=lazy_trials.get(),
                model=settings.model,
                effort=effort,
                max_turns=settings.max_turns,
                country=country,
            )
        except Exception as e:  # never let one run take the worker down
            # The message and traceback can quote patient data (lab values, diagnoses),
            # so neither is logged nor returned: only the exception type and run id.
            error_type = type(e).__name__
            log_event("run_crashed", run_id=run_id, error_type=error_type, status="error")
            result = MatchResult(
                status="error",
                trace=RunTrace(run_id=run_id, patient_id=patient_id, model=settings.model),
                error=f"Internal error during run ({error_type}); see server logs by run_id.",
            )
        wall = round(time.perf_counter() - started, 3)
        jobs.update(run_id, result.status, result)
        usage = result.trace.usage
        log_event(
            "run_finished",
            run_id=run_id,
            agent_run_id=result.trace.run_id,
            patient_id=patient_id,
            status=result.status,
            turns=result.trace.turns,
            tool_calls=len(result.trace.steps),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            grounding_issues=len(result.grounding_issues),
            fallback_used=result.trace.fallback_used,
            wall_seconds=wall,
        )

    @app.get("/healthz", response_model=Health)
    def healthz(request: Request) -> Health:
        s: config.Settings = request.app.state.settings
        return Health(
            model=s.model, http_mode=s.http_mode, api_key_configured=config.credentials_available()
        )

    @app.get("/patients", response_model=list[PatientListing], dependencies=Auth)
    def list_patients(store: StoreDep) -> list[PatientListing]:
        return [patient_listing(store.summary(pid)) for pid in store.list_ids()]

    @app.post("/matches", status_code=202, response_model=MatchAccepted, dependencies=Auth)
    def submit_match(body: MatchRequest, store: StoreDep) -> MatchAccepted:
        try:
            store.summary(body.patient_id)
        except PatientNotFound:
            raise HTTPException(404, f"Unknown patient id {body.patient_id!r}.") from None
        if needs_credentials and not config.credentials_available():
            raise HTTPException(503, "No Anthropic credential configured; set ANTHROPIC_API_KEY.")
        job = jobs.create(body.patient_id, max_active=settings.max_queued_runs)
        if job is None:
            log_event("run_rejected", reason="queue_full", limit=settings.max_queued_runs)
            raise HTTPException(
                429,
                "Too many queued runs; retry later.",
                headers={"Retry-After": "30"},
            )
        effort = body.effort or settings.effort
        executor.submit(execute, job.run_id, body.patient_id, effort, body.country)
        log_event("run_queued", run_id=job.run_id, patient_id=body.patient_id, effort=effort)
        return MatchAccepted(run_id=job.run_id)

    @app.get("/matches/{run_id}", response_model=MatchJob, dependencies=Auth)
    def get_match(run_id: str) -> MatchJob:
        job = jobs.get(run_id)
        if job is None:
            raise HTTPException(404, f"Unknown run id {run_id!r}.")
        return MatchJob(run_id=job.run_id, status=job.status, result=job.result)

    return app


app = create_app()
