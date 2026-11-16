"""Shared in-memory fakes of the ports, so every module can be tested offline."""

from __future__ import annotations

from datetime import date

import pytest

from trialmatch.models import (
    Condition,
    LabResult,
    Medication,
    PatientSummary,
    TrialEligibility,
    TrialHit,
)
from trialmatch.ports import PatientNotFound, TrialNotFound


class FakePatientStore:
    def __init__(self, patients: dict[str, PatientSummary], labs: dict[str, list[LabResult]]):
        self._patients = patients
        self._labs = labs

    def list_ids(self) -> list[str]:
        return sorted(self._patients)

    def summary(self, patient_id: str) -> PatientSummary:
        if patient_id not in self._patients:
            raise PatientNotFound(patient_id)
        return self._patients[patient_id]

    def labs(self, patient_id: str, query: str) -> list[LabResult]:
        if patient_id not in self._patients:
            raise PatientNotFound(patient_id)
        q = query.lower()
        return [
            lab
            for lab in self._labs.get(patient_id, [])
            if q in lab.display.lower() or (lab.code and q in lab.code.lower())
        ]

    def resource_exists(self, patient_id: str, resource_id: str) -> bool:
        p = self._patients.get(patient_id)
        if p is None:
            return False
        ids = {p.resource_id}
        ids |= {c.resource_id for c in p.conditions}
        ids |= {m.resource_id for m in p.medications}
        ids |= {lab.resource_id for lab in p.latest_labs}
        ids |= {pr.resource_id for pr in p.procedures}
        ids |= {lab.resource_id for lab in self._labs.get(patient_id, [])}
        return resource_id in ids


class FakeTrialsSource:
    def __init__(self, hits: list[TrialHit], eligibility: dict[str, TrialEligibility]):
        self._hits = hits
        self._elig = eligibility
        self.calls: list[tuple] = []

    def search(self, condition: str, country: str | None = None, page_size: int = 10):
        self.calls.append(("search", condition, country, page_size))
        return self._hits[:page_size]

    def eligibility(self, nct_id: str) -> TrialEligibility:
        self.calls.append(("eligibility", nct_id))
        if nct_id not in self._elig:
            raise TrialNotFound(nct_id)
        return self._elig[nct_id]


@pytest.fixture
def diabetic_patient() -> PatientSummary:
    return PatientSummary(
        patient_id="test-001",
        resource_id="Patient/test-001",
        sex="female",
        birth_date=date(1970, 5, 1),
        age_years=56,
        as_of=date(2026, 11, 1),
        country="Netherlands",
        conditions=[
            Condition(
                resource_id="Condition/c-t2d",
                code="44054006",
                display="Type 2 diabetes mellitus",
                clinical_status="active",
                onset=date(2015, 3, 1),
            )
        ],
        medications=[
            Medication(
                resource_id="MedicationRequest/m-metformin",
                code="860975",
                display="Metformin 500 MG Oral Tablet",
                status="active",
            )
        ],
        latest_labs=[
            LabResult(
                resource_id="Observation/o-hba1c-2",
                code="4548-4",
                display="Hemoglobin A1c",
                value=8.1,
                unit="%",
                effective=date(2026, 9, 1),
            ),
            LabResult(
                resource_id="Observation/o-egfr-1",
                code="33914-3",
                display="eGFR",
                value=72.0,
                unit="mL/min/1.73m2",
                effective=date(2026, 9, 1),
            ),
        ],
    )


@pytest.fixture
def fake_store(diabetic_patient) -> FakePatientStore:
    history = [
        diabetic_patient.latest_labs[0],
        LabResult(
            resource_id="Observation/o-hba1c-1",
            code="4548-4",
            display="Hemoglobin A1c",
            value=7.6,
            unit="%",
            effective=date(2026, 3, 1),
        ),
        diabetic_patient.latest_labs[1],
    ]
    return FakePatientStore({"test-001": diabetic_patient}, {"test-001": history})


@pytest.fixture
def fake_trials() -> FakeTrialsSource:
    hit = TrialHit(
        nct_id="NCT00000001",
        title="A study of drug X in adults with type 2 diabetes",
        phases=["PHASE3"],
        conditions=["Type 2 Diabetes"],
        countries=["Netherlands"],
        status="RECRUITING",
    )
    elig = TrialEligibility(
        nct_id="NCT00000001",
        title=hit.title,
        criteria_text=(
            "Inclusion Criteria:\n\n* Adults aged 18 to 75 with type 2 diabetes\n"
            "* HbA1c between 7.0% and 10.5%\n\nExclusion Criteria:\n\n"
            "* eGFR below 30 mL/min/1.73m2\n* Pregnancy or breastfeeding"
        ),
        sex="ALL",
        minimum_age="18 Years",
        maximum_age="75 Years",
        healthy_volunteers=False,
        countries=["Netherlands"],
    )
    return FakeTrialsSource([hit], {"NCT00000001": elig})
