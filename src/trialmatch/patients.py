"""Patient store backed by FHIR R4 Bundle files on disk.

Each ``*.json`` file in the directory is one Bundle holding one Patient and that
patient's clinical resources. Both the hand-written ``collection`` bundles in
``data/patients`` and real Synthea output (``transaction`` bundles with ``urn:uuid``
fullUrls) go through the same parser. Bundles without a Patient resource (Synthea's
hospital and practitioner files) are skipped.

Ages are computed against a fixed ``as_of`` date rather than today's date, so that
results, tests and eval runs are reproducible: a patient who is 17 today must still be
17 when the eval runs next month.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from .models import Condition, LabResult, Medication, PatientSummary, Procedure
from .ports import PatientNotFound

DEFAULT_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "patients"
DEFAULT_AS_OF = date(2026, 11, 1)

PREGNANCY_STATUS_LOINC = "82810-3"
_SEX = {"male": "male", "female": "female", "other": "other"}
_SKIP_STATUSES = {"entered-in-error"}
# Condition has no ``status``; a wrong or ruled-out diagnosis is marked here instead.
_SKIP_VERIFICATION = {"entered-in-error", "refuted"}
# Observation results worth reporting. A missing status is accepted for hand-written data;
# cancelled, entered-in-error, registered (no result yet) and unknown are dropped.
_USABLE_OBSERVATION_STATUSES = {"final", "amended", "corrected", "preliminary", None}


# --------------------------------------------------------------------------- helpers


def _parse_date(value: Any) -> date | None:
    """Accept FHIR date or dateTime strings; keep only the calendar date."""
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _first_date(resource: dict, *keys: str) -> date | None:
    for key in keys:
        value = resource.get(key)
        if isinstance(value, dict):  # Period
            value = value.get("start")
        parsed = _parse_date(value)
        if parsed is not None:
            return parsed
    return None


def _first_coding(concept: dict | None) -> dict:
    if not isinstance(concept, dict):
        return {}
    codings = concept.get("coding") or []
    return codings[0] if codings and isinstance(codings[0], dict) else {}


def _concept_code(concept: dict | None) -> str | None:
    return _first_coding(concept).get("code")


def _concept_text(concept: dict | None, default: str) -> str:
    if isinstance(concept, dict) and concept.get("text"):
        return concept["text"]
    return _first_coding(concept).get("display") or default


def _all_codes(concept: dict | None) -> set[str]:
    if not isinstance(concept, dict):
        return set()
    return {c.get("code") for c in concept.get("coding") or [] if c.get("code")}


def _age(birth: date, as_of: date) -> int:
    before_birthday = (as_of.month, as_of.day) < (birth.month, birth.day)
    return as_of.year - birth.year - int(before_birthday)


def _observation_value(obs: dict) -> tuple[float | str | None, str | None]:
    if "valueQuantity" in obs:
        q = obs["valueQuantity"] or {}
        value = q.get("value")
        value = float(value) if isinstance(value, (int, float)) else value
        return value, q.get("unit") or q.get("code")
    if "valueCodeableConcept" in obs:
        return _concept_text(obs["valueCodeableConcept"], "") or None, None
    for key in ("valueString", "valueInteger", "valueBoolean", "valueDateTime"):
        if key in obs:
            v = obs[key]
            if key == "valueInteger":
                return float(v), None
            return str(v).lower() if isinstance(v, bool) else v, None
    return None, None


def _quantity_comparator(obs: dict) -> str | None:
    """``valueQuantity.comparator`` ("<", "<=", ">=", ">"): 15 with ">" means above 15."""
    q = obs.get("valueQuantity")
    comparator = q.get("comparator") if isinstance(q, dict) else None
    return comparator if isinstance(comparator, str) and comparator else None


def _is_usable_observation(obs: dict) -> bool:
    return obs.get("status") in _USABLE_OBSERVATION_STATUSES


def _is_refuted_condition(cond: dict) -> bool:
    return bool(_all_codes(cond.get("verificationStatus")) & _SKIP_VERIFICATION)


def _has_value(obs: dict) -> bool:
    return any(k.startswith("value") for k in obs)


def _is_lab_category(obs: dict) -> bool:
    for cat in obs.get("category") or []:
        if "laboratory" in _all_codes(cat):
            return True
    return False


def _is_pregnancy_status(obs: dict) -> bool:
    return PREGNANCY_STATUS_LOINC in _all_codes(obs.get("code"))


# --------------------------------------------------------------------------- parsing


@dataclass
class _Record:
    patient: dict
    resources: list[dict] = field(default_factory=list)  # everything except the Patient
    ids: set[str] = field(default_factory=set)  # "<ResourceType>/<id>", Patient included
    by_full_url: dict[str, dict] = field(default_factory=dict)

    @property
    def patient_id(self) -> str:
        return self.patient["id"]


def _parse_bundle(bundle: dict, source: str = "<bundle>") -> _Record | None:
    if bundle.get("resourceType") != "Bundle":
        raise ValueError(f"{source}: not a FHIR Bundle")
    patient: dict | None = None
    resources: list[dict] = []
    by_full_url: dict[str, dict] = {}
    for entry in bundle.get("entry") or []:
        resource = entry.get("resource")
        if not isinstance(resource, dict) or "resourceType" not in resource:
            continue
        full_url = entry.get("fullUrl")
        if not resource.get("id") and isinstance(full_url, str):
            # Synthea transaction bundles identify resources by urn:uuid fullUrls.
            resource = {**resource, "id": full_url.rsplit(":", 1)[-1].rsplit("/", 1)[-1]}
        if not resource.get("id"):
            continue
        if full_url:
            by_full_url[full_url] = resource
        if resource["resourceType"] == "Patient":
            if patient is not None:
                raise ValueError(f"{source}: more than one Patient in bundle")
            patient = resource
        else:
            resources.append(resource)
    if patient is None:
        return None
    if _parse_date(patient.get("birthDate")) is None:
        raise ValueError(f"{source}: Patient/{patient['id']} has no valid birthDate")
    ids = {f"Patient/{patient['id']}"} | {f"{r['resourceType']}/{r['id']}" for r in resources}
    return _Record(patient=patient, resources=resources, ids=ids, by_full_url=by_full_url)


# --------------------------------------------------------------------------- store


class FhirPatientStore:
    """``PatientStore`` over a directory of FHIR R4 Bundle JSON files."""

    def __init__(self, directory: str | Path = DEFAULT_DIRECTORY, as_of: date | None = None):
        self.directory = Path(directory)
        self.as_of = as_of or DEFAULT_AS_OF
        self._records: dict[str, _Record] = {}
        if not self.directory.is_dir():
            raise FileNotFoundError(f"patient directory not found: {self.directory}")
        for path in sorted(self.directory.glob("*.json")):
            with path.open(encoding="utf-8") as f:
                record = _parse_bundle(json.load(f), source=str(path))
            if record is None:
                continue
            if record.patient_id in self._records:
                raise ValueError(f"{path}: duplicate patient id {record.patient_id!r}")
            self._records[record.patient_id] = record

    # ------------------------------------------------------------------ protocol

    def list_ids(self) -> list[str]:
        return sorted(self._records)

    def summary(self, patient_id: str) -> PatientSummary:
        rec = self._record(patient_id)
        p = rec.patient
        birth = _parse_date(p.get("birthDate"))
        assert birth is not None  # checked at load time
        addresses = p.get("address") or [{}]
        country = addresses[0].get("country") if isinstance(addresses[0], dict) else None

        conditions: list[Condition] = []
        medications: list[Medication] = []
        procedures: list[Procedure] = []
        pregnancy: list[tuple[date, str, str]] = []
        for r in self._active(rec):
            rtype, rid = r["resourceType"], f"{r['resourceType']}/{r['id']}"
            if rtype == "Condition":
                if _is_refuted_condition(r):
                    continue  # still in the record (resource_exists), but not a diagnosis
                conditions.append(
                    Condition(
                        resource_id=rid,
                        code=_concept_code(r.get("code")),
                        display=_concept_text(r.get("code"), "Unspecified condition"),
                        clinical_status=_concept_code(r.get("clinicalStatus")),
                        onset=_first_date(r, "onsetDateTime", "onsetPeriod", "recordedDate"),
                    )
                )
            elif rtype in ("MedicationRequest", "MedicationStatement"):
                concept = self._medication_concept(rec, r)
                medications.append(
                    Medication(
                        resource_id=rid,
                        code=_concept_code(concept),
                        display=_concept_text(concept, "Unspecified medication"),
                        status=r.get("status"),
                        authored_on=_first_date(
                            r, "authoredOn", "effectiveDateTime", "effectivePeriod"
                        ),
                    )
                )
            elif rtype == "Procedure":
                procedures.append(
                    Procedure(
                        resource_id=rid,
                        code=_concept_code(r.get("code")),
                        display=_concept_text(r.get("code"), "Unspecified procedure"),
                        performed=_first_date(r, "performedDateTime", "performedPeriod"),
                    )
                )
            elif rtype == "Observation" and _is_pregnancy_status(r) and _is_usable_observation(r):
                value, _ = _observation_value(r)
                if isinstance(value, str) and value:
                    when = _first_date(r, "effectiveDateTime", "effectivePeriod", "issued")
                    pregnancy.append((when or date.min, value, rid))

        newest_pregnancy = max(pregnancy) if pregnancy else None

        latest: dict[str, LabResult] = {}
        for lab in self._all_labs(rec):  # newest first, so the first seen per key wins
            key = (lab.code or lab.display).lower()
            latest.setdefault(key, lab)

        return PatientSummary(
            patient_id=rec.patient_id,
            resource_id=f"Patient/{rec.patient_id}",
            sex=_SEX.get(p.get("gender") or "", "unknown"),
            birth_date=birth,
            age_years=_age(birth, self.as_of),
            as_of=self.as_of,
            country=country,
            conditions=conditions,
            medications=medications,
            latest_labs=sorted(latest.values(), key=lambda lab: lab.display.lower()),
            procedures=procedures,
            pregnancy_status=newest_pregnancy[1] if newest_pregnancy else None,
            pregnancy_status_ref=newest_pregnancy[2] if newest_pregnancy else None,
            pregnancy_status_date=(
                newest_pregnancy[0]
                if newest_pregnancy and newest_pregnancy[0] != date.min
                else None
            ),
        )

    def labs(self, patient_id: str, query: str) -> list[LabResult]:
        rec = self._record(patient_id)
        q = query.lower()
        return [
            lab
            for lab in self._all_labs(rec)
            if q in lab.display.lower() or (lab.code and q in lab.code.lower())
        ]

    def resource_exists(self, patient_id: str, resource_id: str) -> bool:
        rec = self._records.get(patient_id)
        return rec is not None and resource_id in rec.ids

    # ------------------------------------------------------------------ internals

    def _record(self, patient_id: str) -> _Record:
        try:
            return self._records[patient_id]
        except KeyError:
            raise PatientNotFound(patient_id) from None

    @staticmethod
    def _active(rec: _Record) -> list[dict]:
        return [r for r in rec.resources if r.get("status") not in _SKIP_STATUSES]

    @staticmethod
    def _medication_concept(rec: _Record, r: dict) -> dict | None:
        if "medicationCodeableConcept" in r:
            return r["medicationCodeableConcept"]
        ref = (r.get("medicationReference") or {}).get("reference")
        if not ref:
            return None
        med = rec.by_full_url.get(ref)
        if med is None:
            med = next((m for m in rec.resources if f"{m['resourceType']}/{m['id']}" == ref), None)
        return med.get("code") if med else None

    def _all_labs(self, rec: _Record) -> list[LabResult]:
        """Every usable Observation with a value (pregnancy status excluded), newest first."""
        out: list[LabResult] = []
        for r in self._active(rec):
            if r["resourceType"] != "Observation" or _is_pregnancy_status(r):
                continue
            if not _is_usable_observation(r):
                continue
            if not (_has_value(r) or _is_lab_category(r)):
                continue
            value, unit = _observation_value(r)
            out.append(
                LabResult(
                    resource_id=f"Observation/{r['id']}",
                    code=_concept_code(r.get("code")),
                    display=_concept_text(r.get("code"), "Unspecified observation"),
                    value=value,
                    comparator=_quantity_comparator(r),
                    unit=unit,
                    effective=_first_date(r, "effectiveDateTime", "effectivePeriod", "issued"),
                )
            )
        # Newest first; undated results last; resource id breaks ties deterministically.
        out.sort(key=lambda lab: lab.resource_id)
        out.sort(key=lambda lab: lab.effective or date.min, reverse=True)
        return out


def load_synthea_directory(path: str | Path, as_of: date | None = None) -> FhirPatientStore:
    """Load real Synthea FHIR R4 output (one transaction Bundle per patient).

    Synthea also writes hospital and practitioner bundles to the same folder; those
    contain no Patient and are skipped by the parser.
    """
    return FhirPatientStore(path, as_of=as_of)
