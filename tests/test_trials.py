"""Offline tests for the ClinicalTrials.gov client (MockTransport + recorded cassettes)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from trialmatch import trials as trials_mod
from trialmatch.models import TrialEligibility, TrialHit
from trialmatch.ports import TrialNotFound, TrialsUnavailable
from trialmatch.trials import CtGovClient, cassette_key

CASSETTES = Path(__file__).parent / "cassettes" / "ctgov"


def _study(nct_id="NCT01234567", criteria="Inclusion Criteria:\n\n* Adults", **elig):
    em = {
        "eligibilityCriteria": criteria,
        "sex": "FEMALE",
        "minimumAge": "18 Years",
        "maximumAge": "75 Years",
        "healthyVolunteers": False,
    }
    em.update(elig)
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id, "briefTitle": "Drug X in T2D"},
            "statusModule": {"overallStatus": "RECRUITING"},
            "conditionsModule": {"conditions": ["Type 2 Diabetes"]},
            "designModule": {"phases": ["PHASE2", "PHASE3"]},
            "eligibilityModule": em,
            "contactsLocationsModule": {
                "locations": [
                    {"country": "United States"},
                    {"country": "Netherlands"},
                    {"country": "United States"},
                    {},
                ]
            },
        }
    }


@pytest.fixture
def sleeps(monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr(trials_mod.time, "sleep", lambda s: calls.append(s))
    return calls


def _client(handler, **kw) -> CtGovClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return CtGovClient(mode="live", http=http, **kw)


# --------------------------------------------------------------------------- parsing


def test_search_parses_and_sends_expected_params(sleeps):
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"studies": [_study()], "nextPageToken": "abc"})

    hits = _client(handler).search("type 2 diabetes", country="Netherlands", page_size=99)
    assert hits == [
        TrialHit(
            nct_id="NCT01234567",
            title="Drug X in T2D",
            phases=["PHASE2", "PHASE3"],
            conditions=["Type 2 Diabetes"],
            countries=["Netherlands", "United States"],
            status="RECRUITING",
        )
    ]
    req = seen[0]
    assert req.url.path == "/api/v2/studies"
    q = req.url.params
    assert q["query.cond"] == "type 2 diabetes"
    assert q["filter.overallStatus"] == "RECRUITING"
    assert q["query.locn"] == "Netherlands"
    assert q["pageSize"] == "25"  # clamped
    assert req.headers["User-Agent"].startswith("trialmatch-agent/")
    assert sleeps == []


def test_search_without_country_and_minimum_page_size():
    seen: list[httpx.Request] = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, json={"studies": []})

    assert _client(handler).search("asthma", page_size=0) == []
    assert "query.locn" not in seen[0].url.params
    assert seen[0].url.params["pageSize"] == "1"


def test_search_handles_missing_optional_modules():
    study = {"protocolSection": {"identificationModule": {"nctId": "NCT07000001"}}}
    hits = _client(lambda r: httpx.Response(200, json={"studies": [study]})).search("x")
    assert hits[0].phases == [] and hits[0].countries == [] and hits[0].title == ""


def test_eligibility_parses_and_unescapes_markdown():
    seen: list[httpx.Request] = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, json=_study(criteria="* HbA1c \\< 10.5% and eGFR \\>= 30"))

    e = _client(handler).eligibility(" nct01234567 ")
    assert seen[0].url.path == "/api/v2/studies/NCT01234567"
    assert isinstance(e, TrialEligibility)
    assert e.nct_id == "NCT01234567"
    assert e.criteria_text == "* HbA1c < 10.5% and eGFR >= 30"
    assert e.sex == "FEMALE"
    assert (e.minimum_age, e.maximum_age, e.healthy_volunteers) == ("18 Years", "75 Years", False)
    assert e.countries == ["Netherlands", "United States"]


def test_eligibility_unknown_sex_and_missing_fields_default():
    study = _study(sex="weird")
    del study["protocolSection"]["eligibilityModule"]["maximumAge"]
    del study["protocolSection"]["eligibilityModule"]["healthyVolunteers"]
    e = _client(lambda r: httpx.Response(200, json=study)).eligibility("NCT01234567")
    assert e.sex == "ALL" and e.maximum_age is None and e.healthy_volunteers is None


def test_long_criteria_are_trimmed_with_marker():
    long_text = "\n".join(f"* criterion number {i} " + "x" * 80 for i in range(300))
    e = _client(lambda r: httpx.Response(200, json=_study(criteria=long_text))).eligibility(
        "NCT01234567"
    )
    assert len(e.criteria_text) < 8200
    assert "[truncated" in e.criteria_text
    assert str(len(long_text)) in e.criteria_text
    assert e.criteria_text.startswith("* criterion number 0 ")


# --------------------------------------------------------------------------- errors / retries


@pytest.mark.parametrize("bad", ["", "NCT123", "NCT123456789", "ABC01234567", "NCT0123456X"])
def test_bad_id_format_raises_not_found_without_request(bad):
    def handler(req):
        raise AssertionError("no request expected")

    with pytest.raises(TrialNotFound):
        _client(handler).eligibility(bad)


def test_404_raises_not_found(sleeps):
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(404, text="NCT number NCT09999999 not found")

    with pytest.raises(TrialNotFound):
        _client(handler).eligibility("NCT09999999")
    assert len(calls) == 1 and sleeps == []


def test_retry_on_503_then_success(sleeps):
    responses = [httpx.Response(503), httpx.Response(502), httpx.Response(200, json=_study())]

    def handler(req):
        return responses.pop(0)

    e = _client(handler).eligibility("NCT01234567")
    assert e.nct_id == "NCT01234567"
    assert sleeps == [0.5, 1.0]  # exponential backoff


def test_retry_on_transport_error(sleeps):
    state = {"n": 0}

    def handler(req):
        state["n"] += 1
        if state["n"] == 1:
            raise httpx.ConnectError("boom", request=req)
        return httpx.Response(200, json={"studies": []})

    assert _client(handler).search("asthma") == []
    assert state["n"] == 2 and len(sleeps) == 1


def test_429_honours_retry_after(sleeps):
    responses = [
        httpx.Response(429, headers={"Retry-After": "7"}),
        httpx.Response(200, json={"studies": [_study()]}),
    ]
    hits = _client(lambda r: responses.pop(0)).search("diabetes")
    assert len(hits) == 1
    assert sleeps == [7.0]


def test_retry_after_is_capped(sleeps):
    responses = [
        httpx.Response(429, headers={"Retry-After": "3600"}),
        httpx.Response(200, json={"studies": []}),
    ]
    _client(lambda r: responses.pop(0)).search("diabetes")
    assert sleeps == [trials_mod.MAX_RETRY_AFTER]


def test_gives_up_after_retries(sleeps):
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(500)

    with pytest.raises(TrialsUnavailable, match="HTTP 500"):
        _client(handler, max_retries=3).search("diabetes")
    assert len(calls) == 4
    assert sleeps == [0.5, 1.0, 2.0]


def test_no_retry_on_400(sleeps):
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(400, text="bad filter")

    with pytest.raises(TrialsUnavailable, match="400"):
        _client(handler).search("diabetes")
    assert len(calls) == 1 and sleeps == []


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        CtGovClient(mode="sometimes")


def test_mode_defaults_from_env(monkeypatch):
    monkeypatch.setenv("TRIALMATCH_HTTP", "replay")
    assert CtGovClient().mode == "replay"
    monkeypatch.delenv("TRIALMATCH_HTTP")
    assert CtGovClient().mode == "live"


# --------------------------------------------------------------------------- record / replay


def test_record_writes_readable_cassette(tmp_path, sleeps):
    http = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"studies": [_study()]}))
    )
    client = CtGovClient(mode="record", cassette_dir=tmp_path, http=http)
    client.search("asthma", country="Netherlands")
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["request"]["path"] == "/api/v2/studies"
    assert data["request"]["params"]["query.locn"] == "Netherlands"
    assert data["response"]["status"] == 200
    expected = cassette_key("GET", "/api/v2/studies", data["request"]["params"])
    assert files[0].stem == expected

    replay = CtGovClient(mode="replay", cassette_dir=tmp_path)
    assert replay.search("asthma", country="Netherlands")[0].nct_id == "NCT01234567"


def test_replay_search_and_eligibility_round_trip():
    client = CtGovClient(mode="replay", cassette_dir=CASSETTES)
    hits = client.search("type 2 diabetes", country="Netherlands", page_size=10)
    assert 0 < len(hits) <= 10
    assert all(h.status == "RECRUITING" for h in hits)
    assert all(h.countries == sorted(set(h.countries)) for h in hits)
    assert any("Netherlands" in h.countries for h in hits)

    e = client.eligibility(hits[0].nct_id)
    assert e.nct_id == hits[0].nct_id
    assert e.title == hits[0].title
    assert "inclusion" in e.criteria_text.lower()
    assert e.sex in ("ALL", "MALE", "FEMALE")


def test_replay_miss_raises_naming_request(tmp_path):
    client = CtGovClient(mode="replay", cassette_dir=tmp_path)
    with pytest.raises(TrialsUnavailable, match="no cassette.*glioblastoma"):
        client.search("glioblastoma", country="Iceland")
    with pytest.raises(TrialsUnavailable, match="NCT01234567"):
        client.eligibility("NCT01234567")


def test_trim_keeps_exclusion_section_of_long_criteria():
    from trialmatch.trials import trim_criteria

    incl = "Inclusion Criteria:\n\n" + "\n".join(f"* inclusion item {i}" for i in range(900))
    excl = "Exclusion Criteria:\n\n* Pregnancy or breastfeeding\n* eGFR below 30"
    out = trim_criteria(incl + "\n\n" + excl, limit=8000)
    assert len(out) < 8300
    assert "Pregnancy or breastfeeding" in out and "eGFR below 30" in out
    assert "[truncated:" in out and "inclusion criteria]" in out
