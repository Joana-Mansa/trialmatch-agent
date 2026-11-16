"""Interfaces between the agent and the outside world.

The agent only sees these protocols, so a deployment can swap the local FHIR files
for a hospital FHIR server, or the live ClinicalTrials.gov client for recorded
responses, without touching agent code.
"""

from __future__ import annotations

from typing import Protocol

from .models import LabResult, PatientSummary, TrialEligibility, TrialHit


class PatientNotFound(KeyError):
    pass


class TrialNotFound(KeyError):
    pass


class TrialsUnavailable(RuntimeError):
    """The trials source failed after retries (network, 5xx, rate limit)."""


class PatientStore(Protocol):
    def list_ids(self) -> list[str]: ...

    def summary(self, patient_id: str) -> PatientSummary:
        """Raises PatientNotFound."""
        ...

    def labs(self, patient_id: str, query: str) -> list[LabResult]:
        """All lab results (full history, newest first) whose display name or code
        contains ``query`` (case-insensitive). Raises PatientNotFound."""
        ...

    def resource_exists(self, patient_id: str, resource_id: str) -> bool:
        """True if ``resource_id`` (e.g. "Observation/obs-1") is in this patient's record."""
        ...


class TrialsSource(Protocol):
    def search(
        self, condition: str, country: str | None = None, page_size: int = 10
    ) -> list[TrialHit]:
        """Recruiting trials only. Raises TrialsUnavailable."""
        ...

    def eligibility(self, nct_id: str) -> TrialEligibility:
        """Raises TrialNotFound or TrialsUnavailable."""
        ...
