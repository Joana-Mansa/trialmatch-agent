"""Runtime settings, read from environment variables, plus factories for the ports.

Variables (all optional):

* ``TRIALMATCH_PATIENT_DIR``: directory of FHIR Bundle files (default: the repo's
  ``data/patients``, or ``./data/patients`` when the package is installed elsewhere).
* ``TRIALMATCH_HTTP``: ClinicalTrials.gov client mode, ``live`` | ``record`` | ``replay``.
* ``TRIALMATCH_PROVIDER``: ``anthropic`` (Claude, default) or ``ollama`` (local model).
* ``TRIALMATCH_MODEL``: model id (default ``claude-opus-5-5`` for anthropic,
  ``qwen3:8b`` for ollama).
* ``TRIALMATCH_OLLAMA_URL``: Ollama server (default ``http://localhost:11434``).
* ``TRIALMATCH_OLLAMA_NUM_CTX``: context window to request from Ollama (default 24576).
* ``TRIALMATCH_EFFORT``: default effort, ``low`` | ``medium`` | ``high`` | ``xhigh`` | ``max``.
* ``TRIALMATCH_MAX_TURNS``: agent turn cap per run (default 25).
* ``TRIALMATCH_MAX_CONCURRENT_RUNS``: API worker threads (default 2).
* ``TRIALMATCH_MAX_QUEUED_RUNS``: cap on queued plus running API jobs; further
  ``POST /matches`` requests get 429 (default 10).
* ``TRIALMATCH_API_TOKEN``: if set, every API endpoint except ``/healthz`` requires
  ``Authorization: Bearer <token>``. If unset, the API is unauthenticated (a warning
  is logged at startup); only bind it to localhost then.

The Anthropic credential is never read here. The SDK reads ``ANTHROPIC_API_KEY`` (or
its other credential sources) itself; this module only reports whether one is present.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .patients import DEFAULT_DIRECTORY, FhirPatientStore
from .trials import MODES, CtGovClient

EFFORTS = ("low", "medium", "high", "xhigh", "max")
PROVIDERS = ("anthropic", "ollama")
DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_MODELS = {"anthropic": DEFAULT_MODEL, "ollama": "qwen3:8b"}


def _default_patient_dir() -> Path:
    # Editable install / source checkout: the repo's data/patients. A regular install
    # (e.g. the Docker image) has no repo around it, so fall back to the cwd.
    if DEFAULT_DIRECTORY.is_dir():
        return DEFAULT_DIRECTORY
    return Path.cwd() / "data" / "patients"


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    patient_dir: Path
    http_mode: str = "live"
    provider: str = "anthropic"
    model: str = DEFAULT_MODEL
    ollama_url: str = "http://localhost:11434"
    ollama_num_ctx: int = 24_576
    effort: str = "high"
    max_turns: int = 25
    max_concurrent_runs: int = 2
    max_queued_runs: int = 10
    api_token: str | None = field(default=None, repr=False)  # never printed

    def run_kwargs(self) -> dict:
        """Provider settings to pass to ``run_match``."""
        return {
            "provider": self.provider,
            "model": self.model,
            "ollama_url": self.ollama_url,
            "ollama_num_ctx": self.ollama_num_ctx,
        }

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        patient_dir = (env.get("TRIALMATCH_PATIENT_DIR") or "").strip()
        http_mode = (env.get("TRIALMATCH_HTTP") or "live").strip().lower()
        if http_mode not in MODES:
            raise ValueError(f"TRIALMATCH_HTTP must be one of {MODES}, got {http_mode!r}")
        effort = (env.get("TRIALMATCH_EFFORT") or "high").strip().lower()
        if effort not in EFFORTS:
            raise ValueError(f"TRIALMATCH_EFFORT must be one of {EFFORTS}, got {effort!r}")
        provider = (env.get("TRIALMATCH_PROVIDER") or "anthropic").strip().lower()
        if provider not in PROVIDERS:
            raise ValueError(f"TRIALMATCH_PROVIDER must be one of {PROVIDERS}, got {provider!r}")
        return cls(
            patient_dir=Path(patient_dir) if patient_dir else _default_patient_dir(),
            http_mode=http_mode,
            provider=provider,
            model=(env.get("TRIALMATCH_MODEL") or DEFAULT_MODELS[provider]).strip(),
            ollama_url=(env.get("TRIALMATCH_OLLAMA_URL") or "http://localhost:11434").strip(),
            ollama_num_ctx=_positive_int(env, "TRIALMATCH_OLLAMA_NUM_CTX", 24_576),
            effort=effort,
            max_turns=_positive_int(env, "TRIALMATCH_MAX_TURNS", 25),
            max_concurrent_runs=_positive_int(env, "TRIALMATCH_MAX_CONCURRENT_RUNS", 2),
            max_queued_runs=_positive_int(env, "TRIALMATCH_MAX_QUEUED_RUNS", 10),
            api_token=(env.get("TRIALMATCH_API_TOKEN") or "").strip() or None,
        )


def get_settings() -> Settings:
    return Settings.from_env()


def make_store(settings: Settings | None = None) -> FhirPatientStore:
    settings = settings or get_settings()
    return FhirPatientStore(settings.patient_dir)


def make_trials(settings: Settings | None = None) -> CtGovClient:
    settings = settings or get_settings()
    return CtGovClient(mode=settings.http_mode)


# --------------------------------------------------------------------------- credentials


def _auto_discoverable_credentials() -> bool:
    """Profile or workload-federation config the SDK would pick up on its own."""
    try:
        from anthropic.lib.credentials._constants import _has_auto_discoverable_credentials
    except Exception:  # private SDK helper; absent in other SDK versions
        return False
    try:
        return bool(_has_auto_discoverable_credentials())
    except Exception:
        return False


def api_key_configured() -> bool:
    """True if ``ANTHROPIC_API_KEY`` is set. Only presence is checked; the value is unused."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def credentials_available() -> bool:
    """True if the Anthropic SDK should find some credential without extra arguments."""
    return (
        api_key_configured()
        or bool(os.environ.get("ANTHROPIC_AUTH_TOKEN"))
        or _auto_discoverable_credentials()
    )


# --------------------------------------------------------------------------- logging


def configure_logging(level: int = logging.INFO) -> None:
    """Give the ``trialmatch`` logger a stderr handler once, if nothing else did."""
    logger = logging.getLogger("trialmatch")
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
