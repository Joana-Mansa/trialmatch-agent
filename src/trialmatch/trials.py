"""ClinicalTrials.gov v2 client implementing the ``TrialsSource`` port.

Three modes, chosen by ``mode`` or the ``TRIALMATCH_HTTP`` env var:

* ``live``: talk to clinicaltrials.gov.
* ``record``: talk to clinicaltrials.gov and save each final response as a cassette.
* ``replay``: serve responses from cassettes only; a missing cassette is an error.

A cassette is ``<cassette_dir>/<sha256 of method + path + sorted params>.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx

from .models import TrialEligibility, TrialHit
from .ports import TrialNotFound, TrialsUnavailable

BASE_URL = "https://clinicaltrials.gov"
STUDIES_PATH = "/api/v2/studies"
USER_AGENT = "trialmatch-agent/0.1"
DEFAULT_CASSETTE_DIR = Path(__file__).resolve().parents[2] / "tests" / "cassettes" / "ctgov"

SEARCH_FIELDS = "NCTId,BriefTitle,Phase,Condition,OverallStatus,LocationCountry"
ELIGIBILITY_FIELDS = "NCTId,BriefTitle,OverallStatus,LocationCountry,EligibilityModule"

MAX_CRITERIA_CHARS = 8000
MAX_RETRY_AFTER = 60.0
NCT_RE = re.compile(r"^NCT\d{8}$")
# CT.gov stores criteria as Markdown with backslash-escaped punctuation ("\<", "\>",
# "\[", "\*", ...). Unescape so "HbA1c \< 7%" reads as "HbA1c < 7%".
_MD_ESCAPE_RE = re.compile(r"\\([!-/:-@\[-`{-~])")

MODES = ("live", "record", "replay")


def cassette_key(method: str, path: str, params: dict[str, Any]) -> str:
    items = sorted((str(k), str(v)) for k, v in params.items())
    raw = method.upper() + " " + path + " " + json.dumps(items, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _countries(ps: dict) -> list[str]:
    locs = (ps.get("contactsLocationsModule") or {}).get("locations") or []
    return sorted({loc["country"].strip() for loc in locs if loc.get("country")})


def _title(ps: dict) -> str:
    ident = ps.get("identificationModule") or {}
    return ident.get("briefTitle") or ident.get("officialTitle") or ""


TRUNCATION_MARKER = "[truncated:"
_EXCLUSION_HEADER_RE = re.compile(r"^\W*exclusion criteria\b", re.I | re.M)


def _cut(text: str, limit: int, label: str) -> str:
    if len(text) <= limit:
        return text
    cut = text.rfind("\n", 0, limit)
    if cut < limit // 2:
        cut = limit
    return (
        text[:cut].rstrip()
        + f"\n{TRUNCATION_MARKER} showing {cut} of {len(text)} characters of {label}]"
    )


def trim_criteria(text: str, limit: int = MAX_CRITERIA_CHARS) -> str:
    """Unescape and trim criteria text, keeping both inclusion and exclusion sections.

    A plain head cut can drop the whole exclusion section of a long trial, which would
    make an excluded patient look eligible. So the budget is split between the two
    sections, and space one section does not need goes to the other.
    """
    text = _MD_ESCAPE_RE.sub(r"\1", text or "").strip()
    if len(text) <= limit:
        return text
    m = _EXCLUSION_HEADER_RE.search(text)
    if not m or m.start() == 0:
        return _cut(text, limit, "eligibility criteria")
    incl, excl = text[: m.start()].rstrip(), text[m.start() :].strip()
    half = limit // 2
    incl_budget = max(half, limit - len(excl))
    excl_budget = max(half, limit - len(incl))
    return (
        _cut(incl, incl_budget, "inclusion criteria")
        + "\n\n"
        + _cut(excl, excl_budget, "exclusion criteria")
    )


def _parse_hit(study: dict) -> TrialHit:
    ps = study.get("protocolSection") or {}
    return TrialHit(
        nct_id=(ps.get("identificationModule") or {}).get("nctId", ""),
        title=_title(ps),
        phases=list((ps.get("designModule") or {}).get("phases") or []),
        conditions=list((ps.get("conditionsModule") or {}).get("conditions") or []),
        countries=_countries(ps),
        status=(ps.get("statusModule") or {}).get("overallStatus") or "UNKNOWN",
    )


def _parse_eligibility(study: dict, nct_id: str) -> TrialEligibility:
    ps = study.get("protocolSection") or {}
    em = ps.get("eligibilityModule") or {}
    sex = str(em.get("sex") or "ALL").upper()
    if sex not in ("ALL", "MALE", "FEMALE"):
        sex = "ALL"
    hv = em.get("healthyVolunteers")
    return TrialEligibility(
        nct_id=(ps.get("identificationModule") or {}).get("nctId") or nct_id,
        title=_title(ps),
        status=(ps.get("statusModule") or {}).get("overallStatus") or "UNKNOWN",
        criteria_text=trim_criteria(em.get("eligibilityCriteria") or ""),
        sex=sex,
        minimum_age=em.get("minimumAge"),
        maximum_age=em.get("maximumAge"),
        healthy_volunteers=hv if isinstance(hv, bool) else None,
        countries=_countries(ps),
    )


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, dt.timestamp() - time.time())


class CtGovClient:
    def __init__(
        self,
        mode: str | None = None,
        cassette_dir: Path | None = None,
        http: httpx.Client | None = None,
        timeout: float = 15.0,
        max_retries: int = 3,
        backoff_base: float = 0.5,
    ) -> None:
        mode = (mode or os.environ.get("TRIALMATCH_HTTP") or "live").strip().lower()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.cassette_dir = Path(cassette_dir) if cassette_dir else DEFAULT_CASSETTE_DIR
        self.max_retries = max(0, max_retries)
        self.backoff_base = backoff_base
        self._owns_http = http is None
        self._http = http or httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT})
        if not self._http.headers.get("User-Agent", "").startswith("trialmatch-agent"):
            # Replace httpx's default "python-httpx/x.y" with our generic agent string.
            self._http.headers["User-Agent"] = USER_AGENT

    # ----------------------------------------------------------------- public API

    def search(
        self, condition: str, country: str | None = None, page_size: int = 10
    ) -> list[TrialHit]:
        condition = (condition or "").strip()
        if not condition:
            raise ValueError("condition must be a non-empty string")
        params: dict[str, Any] = {
            "query.cond": condition,
            "filter.overallStatus": "RECRUITING",
            "pageSize": max(1, min(25, int(page_size))),
            "fields": SEARCH_FIELDS,
        }
        if country and country.strip():
            params["query.locn"] = country.strip()
        status, body = self._get(STUDIES_PATH, params)
        if status != 200 or not isinstance(body, dict):
            raise TrialsUnavailable(f"ClinicalTrials.gov search failed with HTTP {status}")
        hits = [_parse_hit(s) for s in body.get("studies") or []]
        return [h for h in hits if h.nct_id]

    def eligibility(self, nct_id: str) -> TrialEligibility:
        nid = (nct_id or "").strip().upper()
        if not NCT_RE.match(nid):
            raise TrialNotFound(f"invalid NCT id {nct_id!r}: expected 'NCT' followed by 8 digits")
        status, body = self._get(f"{STUDIES_PATH}/{nid}", {"fields": ELIGIBILITY_FIELDS})
        if status == 404:
            raise TrialNotFound(f"{nid} not found on ClinicalTrials.gov")
        if status != 200 or not isinstance(body, dict):
            raise TrialsUnavailable(f"ClinicalTrials.gov returned HTTP {status} for {nid}")
        return _parse_eligibility(body, nid)

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> CtGovClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ----------------------------------------------------------------- transport

    def _cassette_path(self, path: str, params: dict[str, Any]) -> Path:
        return self.cassette_dir / f"{cassette_key('GET', path, params)}.json"

    def _get(self, path: str, params: dict[str, Any]) -> tuple[int, Any]:
        params = {k: str(v) for k, v in params.items()}
        if self.mode == "replay":
            cpath = self._cassette_path(path, params)
            if not cpath.exists():
                raise TrialsUnavailable(
                    f"replay mode: no cassette for GET {path} params={params} "
                    f"(expected {cpath.name}); re-record with TRIALMATCH_HTTP=record"
                )
            data = json.loads(cpath.read_text(encoding="utf-8"))
            return int(data["response"]["status"]), data["response"]["json"]

        status, body = self._get_live(path, params)
        if self.mode == "record":
            self._write_cassette(path, params, status, body)
        return status, body

    def _write_cassette(self, path: str, params: dict[str, str], status: int, body: Any) -> None:
        self.cassette_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "request": {"method": "GET", "path": path, "params": dict(sorted(params.items()))},
            "response": {"status": status, "json": body},
        }
        text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        self._cassette_path(path, params).write_text(text, encoding="utf-8")

    def _get_live(self, path: str, params: dict[str, str]) -> tuple[int, Any]:
        last_error = ""
        for attempt in range(self.max_retries + 1):
            retry_after: float | None = None
            try:
                resp = self._http.get(BASE_URL + path, params=params)
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                status = resp.status_code
                if status == 429 or status >= 500:
                    last_error = f"HTTP {status}"
                    retry_after = _retry_after_seconds(resp.headers.get("Retry-After"))
                elif status == 404:
                    return status, None
                elif status >= 400:
                    raise TrialsUnavailable(
                        f"ClinicalTrials.gov rejected GET {path}: HTTP {status} {resp.text[:200]}"
                    )
                else:
                    try:
                        return status, resp.json()
                    except ValueError as exc:
                        raise TrialsUnavailable(
                            f"ClinicalTrials.gov returned non-JSON for GET {path}"
                        ) from exc
            if attempt < self.max_retries:
                delay = self.backoff_base * (2**attempt)
                if retry_after is not None:
                    delay = max(delay, min(retry_after, MAX_RETRY_AFTER))
                time.sleep(delay)
        raise TrialsUnavailable(
            f"ClinicalTrials.gov unavailable after {self.max_retries + 1} attempts "
            f"(GET {path}): {last_error}"
        )
