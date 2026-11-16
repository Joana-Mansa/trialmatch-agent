from __future__ import annotations

import json
from datetime import date

import pytest

from trialmatch.patients import FhirPatientStore, load_synthea_directory
from trialmatch.ports import PatientNotFound, PatientStore

ALL_IDS = [f"p00{i}" for i in range(1, 9)]


@pytest.fixture(scope="module")
def store() -> FhirPatientStore:
    return FhirPatientStore()


def test_satisfies_protocol(store):
    s: PatientStore = store
    assert s.list_ids() == ALL_IDS


def test_all_patients_load_with_core_fields(store):
    for pid in ALL_IDS:
        s = store.summary(pid)
        assert s.patient_id == pid
        assert s.resource_id == f"Patient/{pid}"
        assert s.as_of == date(2026, 11, 1)
        assert s.country
        assert s.conditions and s.medications and s.latest_labs


@pytest.mark.parametrize(
    "pid,age,sex,country",
    [
        ("p001", 56, "female", "Netherlands"),
        ("p002", 68, "male", "United States"),
        ("p003", 47, "female", "United States"),
        ("p004", 34, "female", "Netherlands"),
        ("p005", 72, "male", "Netherlands"),
        ("p006", 81, "male", "United States"),
        ("p007", 17, "female", "Netherlands"),
        ("p008", 52, "female", "Germany"),
    ],
)
def test_demographics(store, pid, age, sex, country):
    s = store.summary(pid)
    assert (s.age_years, s.sex, s.country) == (age, sex, country)


def test_age_is_birthday_correct():
    # p007 is born 2008-11-20: still 17 the day before, 18 on the birthday.
    assert FhirPatientStore(as_of=date(2026, 11, 19)).summary("p007").age_years == 17
    assert FhirPatientStore(as_of=date(2026, 11, 20)).summary("p007").age_years == 18


def test_p001_clean_diabetic(store):
    s = store.summary("p001")
    assert any(c.code == "44054006" and c.clinical_status == "active" for c in s.conditions)
    latest = {lab.code: lab for lab in s.latest_labs}
    assert latest["4548-4"].value == pytest.approx(8.1)
    assert latest["33914-3"].value == pytest.approx(72)
    assert s.pregnancy_status is None


def test_p002_latest_egfr_below_30(store):
    s = store.summary("p002")
    egfr = next(lab for lab in s.latest_labs if lab.code == "33914-3")
    assert egfr.value < 30
    assert egfr.effective == date(2026, 8, 27)
    assert len(store.labs("p002", "egfr")) >= 2


def test_p003_completed_chemo(store):
    s = store.summary("p003")
    completed = {m.code for m in s.medications if m.status == "completed"}
    assert {"3639", "3002"} <= completed  # doxorubicin, cyclophosphamide
    chemo = [p for p in s.procedures if p.code == "367336001"]
    assert chemo and chemo[0].performed.year == 2024


def test_p004_pregnancy_status(store):
    s = store.summary("p004")
    assert s.pregnancy_status == "Pregnant"
    # The pregnancy status observation is not reported as a lab.
    assert all(lab.code != "82810-3" for lab in s.latest_labs)
    assert store.labs("p004", "82810-3") == []


def test_labs_newest_first_and_filtered(store):
    hba1c = store.labs("p001", "a1C")  # case-insensitive
    assert [lab.value for lab in hba1c] == [8.1, 7.8, 7.4]
    dates = [lab.effective for lab in hba1c]
    assert dates == sorted(dates, reverse=True)
    assert all("a1c" in lab.display.lower() for lab in hba1c)
    by_code = store.labs("p001", "4548-4")
    assert [lab.resource_id for lab in by_code] == [lab.resource_id for lab in hba1c]
    assert store.labs("p001", "no-such-lab") == []


def test_latest_labs_one_per_code(store):
    s = store.summary("p002")
    keys = [lab.code or lab.display for lab in s.latest_labs]
    assert len(keys) == len(set(keys))


def test_resource_exists(store):
    assert store.resource_exists("p001", "Patient/p001")
    assert store.resource_exists("p001", "Condition/p001-cond-1")
    assert not store.resource_exists("p001", "Condition/p002-cond-1")
    assert not store.resource_exists("p001", "Observation/missing")
    assert not store.resource_exists("nobody", "Patient/p001")


def test_every_summary_ref_exists(store):
    for pid in ALL_IDS:
        s = store.summary(pid)
        refs = [s.resource_id]
        refs += [c.resource_id for c in s.conditions]
        refs += [m.resource_id for m in s.medications]
        refs += [lab.resource_id for lab in s.latest_labs]
        refs += [p.resource_id for p in s.procedures]
        refs += [lab.resource_id for lab in store.labs(pid, "")]
        for ref in refs:
            assert store.resource_exists(pid, ref), (pid, ref)


def test_unknown_patient_raises(store):
    with pytest.raises(PatientNotFound):
        store.summary("p999")
    with pytest.raises(PatientNotFound):
        store.labs("p999", "egfr")


def test_bundles_are_tagged_synthetic(store):
    for path in sorted(store.directory.glob("*.json")):
        bundle = json.loads(path.read_text(encoding="utf-8"))
        assert bundle["type"] == "collection"
        codes = {t["code"] for t in bundle["meta"]["tag"]}
        assert "synthetic" in codes
        ids = [f"{e['resource']['resourceType']}/{e['resource']['id']}" for e in bundle["entry"]]
        assert len(ids) == len(set(ids)), path.name


def test_load_synthea_transaction_bundle(tmp_path):
    pid = "5f0c1e2a-0000-4000-8000-000000000001"
    obs = "5f0c1e2a-0000-4000-8000-000000000003"
    med = "5f0c1e2a-0000-4000-8000-000000000004"
    bundle = {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [
            {
                "fullUrl": f"urn:uuid:{pid}",
                "resource": {
                    "resourceType": "Patient",
                    "id": pid,
                    "gender": "male",
                    "birthDate": "1960-12-01",
                    "address": [{"country": "US"}],
                },
            },
            {
                # No id: must be derived from the urn:uuid fullUrl.
                "fullUrl": "urn:uuid:5f0c1e2a-0000-4000-8000-000000000002",
                "resource": {
                    "resourceType": "Condition",
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                    "code": {"coding": [{"code": "44054006", "display": "Diabetes"}]},
                    "subject": {"reference": f"urn:uuid:{pid}"},
                    "onsetDateTime": "2010-01-01T08:00:00-05:00",
                },
            },
            {
                "fullUrl": f"urn:uuid:{obs}",
                "resource": {
                    "resourceType": "Observation",
                    "id": obs,
                    "status": "final",
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "code": {"coding": [{"code": "4548-4", "display": "Hemoglobin A1c"}]},
                    "effectiveDateTime": "2026-01-05T08:00:00-05:00",
                    "valueQuantity": {"value": 7, "unit": "%"},
                },
            },
            {
                "fullUrl": f"urn:uuid:{med}",
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": med,
                    "status": "active",
                    "medicationCodeableConcept": {
                        "coding": [{"code": "6809"}],
                        "text": "Metformin",
                    },
                },
            },
        ],
    }
    (tmp_path / "patient.json").write_text(json.dumps(bundle), encoding="utf-8")
    hospital = {"resourceType": "Bundle", "type": "batch", "entry": []}
    (tmp_path / "hospitalInformation1.json").write_text(json.dumps(hospital), encoding="utf-8")

    store = load_synthea_directory(tmp_path)
    assert store.list_ids() == [pid]
    s = store.summary(pid)
    assert s.age_years == 65 and s.sex == "male" and s.country == "US"
    cond = s.conditions[0]
    assert cond.resource_id == "Condition/5f0c1e2a-0000-4000-8000-000000000002"
    assert cond.onset == date(2010, 1, 1) and cond.display == "Diabetes"
    assert s.latest_labs[0].value == 7.0 and s.latest_labs[0].effective == date(2026, 1, 5)
    assert s.medications[0].display == "Metformin" and s.medications[0].authored_on is None
    assert store.resource_exists(pid, cond.resource_id)
    assert store.resource_exists(pid, f"MedicationRequest/{med}")


# --------------------------------------------------------------------------- review fixes


def _store_from(tmp_path, resources: list[dict]) -> FhirPatientStore:
    patient = {"resourceType": "Patient", "id": "t1", "gender": "female", "birthDate": "1980-01-01"}
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"resource": r} for r in [patient, *resources]],
    }
    (tmp_path / "t1.json").write_text(json.dumps(bundle), encoding="utf-8")
    return FhirPatientStore(tmp_path)


def _condition(rid: str, verification: str | None) -> dict:
    r = {
        "resourceType": "Condition",
        "id": rid,
        "clinicalStatus": {"coding": [{"code": "active"}]},
        "code": {"coding": [{"code": rid, "display": rid}]},
    }
    if verification:
        r["verificationStatus"] = {"coding": [{"code": verification}]}
    return r


def _lab(rid: str, status: str | None, value: dict, when: str = "2026-01-01") -> dict:
    r = {
        "resourceType": "Observation",
        "id": rid,
        "category": [{"coding": [{"code": "laboratory"}]}],
        "code": {"coding": [{"code": "33914-3", "display": "eGFR"}]},
        "effectiveDateTime": when,
        "valueQuantity": value,
    }
    if status:
        r["status"] = status
    return r


def test_refuted_and_erroneous_conditions_excluded(tmp_path):
    store = _store_from(
        tmp_path,
        [
            _condition("confirmed", "confirmed"),
            _condition("unverified", None),
            _condition("refuted", "refuted"),
            _condition("wrong", "entered-in-error"),
        ],
    )
    s = store.summary("t1")
    assert sorted(c.code for c in s.conditions) == ["confirmed", "unverified"]
    # Still part of the record, so a citation of them is not a fabricated id.
    assert store.resource_exists("t1", "Condition/refuted")
    assert store.resource_exists("t1", "Condition/wrong")


def test_quantity_comparator_kept(tmp_path):
    store = _store_from(
        tmp_path, [_lab("egfr", "final", {"value": 15, "comparator": ">", "unit": "mL/min"})]
    )
    lab = store.summary("t1").latest_labs[0]
    assert (lab.value, lab.comparator, lab.unit) == (15.0, ">", "mL/min")
    assert store.labs("t1", "egfr")[0].comparator == ">"


def test_plain_quantity_has_no_comparator(store):
    assert all(lab.comparator is None for lab in store.summary("p001").latest_labs)


@pytest.mark.parametrize("status", ["cancelled", "entered-in-error", "registered", "unknown"])
def test_unusable_observation_statuses_excluded(tmp_path, status):
    store = _store_from(
        tmp_path,
        [
            _lab("good", "final", {"value": 60}, when="2026-01-01"),
            _lab("bad", status, {"value": 10}, when="2026-06-01"),  # newer, but unusable
        ],
    )
    s = store.summary("t1")
    assert [lab.resource_id for lab in s.latest_labs] == ["Observation/good"]
    assert [lab.resource_id for lab in store.labs("t1", "egfr")] == ["Observation/good"]
    assert store.resource_exists("t1", "Observation/bad")


@pytest.mark.parametrize("status", ["final", "amended", "corrected", "preliminary", None])
def test_usable_observation_statuses_kept(tmp_path, status):
    store = _store_from(tmp_path, [_lab("egfr", status, {"value": 45})])
    assert [lab.value for lab in store.summary("t1").latest_labs] == [45.0]


def _pregnancy(rid: str, value: str, when: str | None, status: str = "final") -> dict:
    r = {
        "resourceType": "Observation",
        "id": rid,
        "status": status,
        "code": {"coding": [{"system": "http://loinc.org", "code": "82810-3"}]},
        "valueCodeableConcept": {"text": value},
    }
    if when:
        r["effectiveDateTime"] = when
    return r


def test_pregnancy_status_date(tmp_path):
    store = _store_from(
        tmp_path,
        [
            _pregnancy("preg-old", "Not pregnant", "2025-03-01"),
            _pregnancy("preg-new", "Pregnant", "2026-09-15T10:00:00Z"),
            _pregnancy("preg-void", "Not pregnant", "2026-10-01", status="entered-in-error"),
        ],
    )
    s = store.summary("t1")
    assert s.pregnancy_status == "Pregnant"
    assert s.pregnancy_status_ref == "Observation/preg-new"
    assert s.pregnancy_status_date == date(2026, 9, 15)


def test_pregnancy_status_undated(tmp_path):
    s = _store_from(tmp_path, [_pregnancy("preg", "Pregnant", None)]).summary("t1")
    assert s.pregnancy_status == "Pregnant" and s.pregnancy_status_date is None


def test_p004_pregnancy_status_has_date(store):
    s = store.summary("p004")
    assert s.pregnancy_status_date is not None and s.pregnancy_status_date <= s.as_of
