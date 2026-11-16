"""Evaluation harness: run the agent over the synthetic patients and score the runs.

Every metric is computed in code from the ``MatchResult`` objects, the patient store and
the trials source. There is no LLM judge. Usage::

    python -m trialmatch.eval --patients all --http live --yes

Metrics (all pure functions over ``MatchResult``, store, trials and expectations):

1. run health: status counts
2. hard-constraint violations: a code hard check failed but the verdict is not
   ``likely_ineligible`` (target 0)
3. citation validity: share of evidence refs that exist in the patient record
4. criterion coverage: assessments per match, share unknown, share with evidence
5. planted-disqualifier recall, driven by ``evals/expectations.json``
6. cost and latency from the run trace
7. verdict agreement with optional human gold labels in ``evals/gold.jsonl``
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hard_checks import check
from .models import MatchResult, RunTrace, TrialEligibility, Usage
from .ports import PatientStore, TrialsSource

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPECTATIONS = ROOT / "evals" / "expectations.json"
DEFAULT_GOLD = ROOT / "evals" / "gold.jsonl"
DEFAULT_OUT = ROOT / "evals" / "results"

# Prices per million tokens, USD, taken 2026-11. Cache writes are 1.25x input (5 min TTL).
PRICING: dict[str, Any] = {
    "taken": "2026-11",
    "currency": "USD",
    "per_mtok": {
        "claude-opus-5-5": {
            "input": 4.00,
            "output": 20.00,
            "cache_read": 0.20,
            "cache_write": 4.00 * 1.25,
        },
    },
}


# --------------------------------------------------------------------------- helpers


class CachedTrials:
    """Wraps a ``TrialsSource`` so each trial's eligibility is fetched once per eval.

    Lookups go through the same source the agent used, so in replay mode the text is
    exactly what the agent saw. Failures are remembered and reported, not raised.
    """

    def __init__(self, trials: TrialsSource):
        self._trials = trials
        self._cache: dict[str, TrialEligibility | None] = {}
        self.errors: dict[str, str] = {}
        self._lock = threading.Lock()

    def eligibility(self, nct_id: str) -> TrialEligibility | None:
        key = (nct_id or "").strip().upper()
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        try:
            elig: TrialEligibility | None = self._trials.eligibility(key)
        except Exception as e:  # noqa: BLE001 - any failure means "could not verify"
            elig = None
            with self._lock:
                self.errors[key] = f"{type(e).__name__}: {e}"
        with self._lock:
            self._cache[key] = elig
        return elig


def _as_cached(trials: TrialsSource | CachedTrials) -> CachedTrials:
    return trials if isinstance(trials, CachedTrials) else CachedTrials(trials)


def _matches(result: MatchResult):
    return result.report.matches if result.report is not None else []


def _share(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _mean(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


# --------------------------------------------------------------------------- 1. health


def run_health(results: Iterable[MatchResult]) -> dict[str, int]:
    counts = {"completed": 0, "completed_with_issues": 0, "refused": 0, "incomplete": 0, "error": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    counts["total"] = sum(counts.values())
    return counts


# --------------------------------------------------------------------------- 2. hard checks


def hard_constraint_violations(
    result: MatchResult, store: PatientStore, trials: TrialsSource | CachedTrials
) -> dict[str, Any]:
    """Matches whose code hard check failed but whose verdict is not likely_ineligible.

    Two sources are combined: the run's own grounding issues, and a recomputation of
    ``hard_checks.check`` on freshly fetched eligibility (catches a match the agent
    reported without ever fetching its eligibility).
    """
    cached = _as_cached(trials)
    from_grounding = {
        i.nct_id for i in result.grounding_issues if i.kind == "hard_check_contradiction"
    }
    violations: list[dict[str, Any]] = []
    unverified: list[str] = []
    checked = 0
    try:
        patient = store.summary(result.trace.patient_id)
    except Exception:  # noqa: BLE001
        patient = None

    for m in _matches(result):
        failed: list[str] = []
        recomputed_fail = False
        elig = cached.eligibility(m.nct_id) if patient is not None else None
        if elig is None:
            unverified.append(m.nct_id)
        else:
            checked += 1
            hard = check(patient, elig)
            if not hard.passed:
                failed = sorted(k for k, v in hard.checks.items() if v == "fail")
                recomputed_fail = m.verdict != "likely_ineligible"
        if recomputed_fail or m.nct_id in from_grounding:
            sources = [s for s, hit in (("grounding", m.nct_id in from_grounding),
                                         ("recomputed", recomputed_fail)) if hit]
            violations.append(
                {"nct_id": m.nct_id, "verdict": m.verdict, "failed_checks": failed,
                 "sources": sources}
            )
    return {"checked": checked, "violations": violations, "count": len(violations),
            "unverified": unverified}


# --------------------------------------------------------------------------- 3. citations


def citation_validity(result: MatchResult, store: PatientStore) -> dict[str, Any]:
    """Every evidence ref occurrence is checked against the patient's record."""
    pid = result.trace.patient_id
    total = valid = 0
    fabricated: list[dict[str, str]] = []
    for m in _matches(result):
        for a in m.assessments:
            for ref in a.evidence_refs:
                total += 1
                if store.resource_exists(pid, ref):
                    valid += 1
                else:
                    fabricated.append({"nct_id": m.nct_id, "ref": ref})
    return {"total_refs": total, "valid_refs": valid, "fabricated_count": len(fabricated),
            "fabricated": fabricated, "share_valid": _share(valid, total)}


# --------------------------------------------------------------------------- 4. coverage


def criterion_coverage(result: MatchResult) -> dict[str, Any]:
    per_match = []
    for m in _matches(result):
        n = len(m.assessments)
        unknown = sum(a.status == "unknown" for a in m.assessments)
        with_ev = sum(bool(a.evidence_refs) for a in m.assessments)
        per_match.append(
            {"nct_id": m.nct_id, "verdict": m.verdict, "assessments": n,
             "unknown_share": _share(unknown, n), "evidence_share": _share(with_ev, n)}
        )
    return {
        "matches": len(per_match),
        "per_match": per_match,
        "mean_assessments": _mean(p["assessments"] for p in per_match),
        "mean_unknown_share": _mean(p["unknown_share"] for p in per_match),
        "mean_evidence_share": _mean(p["evidence_share"] for p in per_match),
    }


# --------------------------------------------------------------------------- 5. recall


def load_expectations(path: str | Path = DEFAULT_EXPECTATIONS) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("rules", {})


def planted_recall(
    result: MatchResult,
    store: PatientStore,
    trials: TrialsSource | CachedTrials,
    expectations: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Did the agent catch the patient's planted disqualifier on the trials where it applies?

    Returns None for patients without a rule. ``recall`` is None ("n/a") when no
    assessed trial is relevant to the planted issue.
    """
    pid = result.trace.patient_id
    rule = expectations.get(pid)
    if rule is None:
        return None
    cached = _as_cached(trials)
    kind = rule.get("kind", "criterion")
    out: dict[str, Any] = {"patient_id": pid, "issue": rule.get("issue", ""), "kind": kind,
                           "relevant": 0, "hits": 0, "recall": None, "trials": [],
                           "unverified": []}
    if result.report is None:
        out["reason"] = "no report"
        return out

    if kind == "hard_check":
        try:
            patient = store.summary(pid)
        except Exception:  # noqa: BLE001
            out["reason"] = "patient not found"
            return out
        for m in result.report.matches:
            elig = cached.eligibility(m.nct_id)
            if elig is None:
                out["unverified"].append(m.nct_id)
                continue
            hard = check(patient, elig)
            if hard.passed:
                continue
            hit = m.verdict == "likely_ineligible"
            out["relevant"] += 1
            out["hits"] += hit
            out["trials"].append({"nct_id": m.nct_id, "verdict": m.verdict, "hit": hit,
                                  "detail": " ".join(hard.notes)})
    else:
        pattern = re.compile(rule["criteria_regex"], re.IGNORECASE)
        correct = set(rule.get("correct_refs", []))
        for m in result.report.matches:
            elig = cached.eligibility(m.nct_id)
            if elig is None:
                out["unverified"].append(m.nct_id)
                continue
            found = pattern.search(elig.criteria_text or "")
            if not found:
                continue
            flagged = [
                a for a in m.assessments
                if ((a.kind == "exclusion" and a.status == "met")
                    or (a.kind == "inclusion" and a.status == "not_met"))
                and correct.intersection(a.evidence_refs)
            ]
            verdict_ok = m.verdict == "likely_ineligible"
            hit = bool(flagged) and verdict_ok
            if not flagged:
                why = "no exclusion met / inclusion not_met citing a correct ref"
            elif not verdict_ok:
                why = f"flagged but verdict {m.verdict}"
            else:
                why = f"flagged: {flagged[0].criterion[:80]}"
            out["relevant"] += 1
            out["hits"] += hit
            out["trials"].append({"nct_id": m.nct_id, "verdict": m.verdict, "hit": hit,
                                  "matched_text": found.group(0), "detail": why})
    out["recall"] = _share(out["hits"], out["relevant"])
    if not out["relevant"]:
        out["reason"] = "no assessed trial has a relevant criterion"
    return out


# --------------------------------------------------------------------------- 6. cost


def usage_cost(usage: Usage, model: str, pricing: Mapping[str, Any] = PRICING) -> float | None:
    """Dollar cost of a run's usage; None when the model has no price entry.

    ``input_tokens`` from the API excludes cached tokens, so the four buckets add up.
    Local models (``ollama:...``) have no API cost.
    """
    if model.startswith("ollama:"):
        return 0.0
    p = pricing["per_mtok"].get(model)
    if p is None:
        return None
    dollars = (
        usage.input_tokens * p["input"]
        + usage.output_tokens * p["output"]
        + usage.cache_read_input_tokens * p["cache_read"]
        + usage.cache_creation_input_tokens * p["cache_write"]
    ) / 1_000_000
    return round(dollars, 6)


def cost_latency(result: MatchResult, pricing: Mapping[str, Any] = PRICING) -> dict[str, Any]:
    t = result.trace
    return {
        "input_tokens": t.usage.input_tokens,
        "output_tokens": t.usage.output_tokens,
        "cache_read_input_tokens": t.usage.cache_read_input_tokens,
        "cache_creation_input_tokens": t.usage.cache_creation_input_tokens,
        "cost_usd": usage_cost(t.usage, t.model, pricing),
        "wall_seconds": t.wall_seconds,
        "turns": t.turns,
        "tool_calls": dict(Counter(s.tool for s in t.steps)),
        "tool_errors": sum(not s.ok for s in t.steps),
        "fallback_used": t.fallback_used,
    }


# --------------------------------------------------------------------------- 7. gold


def load_gold(path: str | Path = DEFAULT_GOLD) -> list[dict[str, Any]]:
    """Read gold rows, skipping blanks, ``#`` comments and rows marked placeholder."""
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        if row.get("placeholder") or not row.get("patient_id") or not row.get("nct_id"):
            continue
        if row.get("expected_verdict") not in (
            "likely_eligible", "possibly_eligible", "likely_ineligible"
        ):
            continue
        rows.append(row)
    return rows


def gold_agreement(results: Iterable[MatchResult], gold: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts: dict[tuple[str, str], str] = {}
    for r in results:
        for m in _matches(r):
            verdicts[(r.trace.patient_id, m.nct_id.upper())] = m.verdict
    rows = []
    for g in gold:
        got = verdicts.get((g["patient_id"], g["nct_id"].upper()))
        if got is None:
            continue
        rows.append({"patient_id": g["patient_id"], "nct_id": g["nct_id"],
                     "expected": g["expected_verdict"], "got": got,
                     "agree": got == g["expected_verdict"]})
    agreed = sum(r["agree"] for r in rows)
    return {"gold_rows": len(gold), "compared": len(rows), "agreed": agreed,
            "agreement": _share(agreed, len(rows)), "rows": rows}


# --------------------------------------------------------------------------- aggregate


def evaluate_result(
    result: MatchResult,
    store: PatientStore,
    trials: TrialsSource | CachedTrials,
    expectations: Mapping[str, Any],
    pricing: Mapping[str, Any] = PRICING,
) -> dict[str, Any]:
    cached = _as_cached(trials)
    return {
        "patient_id": result.trace.patient_id,
        "status": result.status,
        "error": result.error,
        "matches": len(_matches(result)),
        "grounding_issues": Counter(i.kind for i in result.grounding_issues),
        "hard_constraints": hard_constraint_violations(result, store, cached),
        "citations": citation_validity(result, store),
        "coverage": criterion_coverage(result),
        "planted": planted_recall(result, store, cached, expectations),
        "cost": cost_latency(result, pricing),
    }


def summarize(
    results: list[MatchResult],
    store: PatientStore,
    trials: TrialsSource | CachedTrials,
    expectations: Mapping[str, Any],
    gold: list[dict[str, Any]] | None = None,
    pricing: Mapping[str, Any] = PRICING,
) -> dict[str, Any]:
    cached = _as_cached(trials)
    per_patient = [evaluate_result(r, store, cached, expectations, pricing) for r in results]
    for p in per_patient:
        p["grounding_issues"] = dict(p["grounding_issues"])

    total_refs = sum(p["citations"]["total_refs"] for p in per_patient)
    valid_refs = sum(p["citations"]["valid_refs"] for p in per_patient)
    planted = [p["planted"] for p in per_patient if p["planted"] is not None]
    relevant = sum(x["relevant"] for x in planted)
    hits = sum(x["hits"] for x in planted)
    costs = [p["cost"]["cost_usd"] for p in per_patient]
    tool_calls: Counter[str] = Counter()
    for p in per_patient:
        tool_calls.update(p["cost"]["tool_calls"])
    all_matches = [m for p in per_patient for m in p["coverage"]["per_match"]]

    overall = {
        "health": run_health(results),
        "hard_constraint_violations": sum(p["hard_constraints"]["count"] for p in per_patient),
        "matches": len(all_matches),
        "citations": {"total_refs": total_refs, "valid_refs": valid_refs,
                      "fabricated_count": total_refs - valid_refs,
                      "share_valid": _share(valid_refs, total_refs)},
        "coverage": {
            "mean_assessments": _mean(m["assessments"] for m in all_matches),
            "mean_unknown_share": _mean(m["unknown_share"] for m in all_matches),
            "mean_evidence_share": _mean(m["evidence_share"] for m in all_matches),
        },
        "planted_recall": {"relevant": relevant, "hits": hits,
                           "recall": _share(hits, relevant)},
        "cost": {
            "total_usd": round(sum(c for c in costs if c is not None), 6),
            "unpriced_runs": sum(c is None for c in costs),
            "mean_wall_seconds": _mean(p["cost"]["wall_seconds"] for p in per_patient),
            "total_turns": sum(p["cost"]["turns"] for p in per_patient),
            "tool_calls": dict(tool_calls),
            "fallback_runs": sum(p["cost"]["fallback_used"] for p in per_patient),
        },
        "gold": gold_agreement(results, gold or []),
        "eligibility_fetch_errors": dict(cached.errors),
    }
    return {"overall": overall, "patients": per_patient, "pricing": dict(pricing)}


# --------------------------------------------------------------------------- markdown


def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def _fmt(x: Any, nd: int = 2) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    def cell(v: Any) -> str:
        return str(v).replace("|", "/").replace("\n", " ")

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return lines


def render_markdown(summary: Mapping[str, Any], meta: Mapping[str, Any] | None = None) -> str:
    o = summary["overall"]
    h = o["health"]
    meta = meta or {}
    lines = ["# trialmatch-agent eval report", ""]
    if meta:
        lines += [f"- {k}: {v}" for k, v in meta.items()] + [""]

    lines += ["## Summary", ""]
    pr = o["planted_recall"]
    rows = [
        ["Runs completed", f"{h['completed']} of {h['total']} (refused {h['refused']}, "
         f"incomplete {h['incomplete']}, error {h['error']})", "all"],
        ["Hard-constraint violations", o["hard_constraint_violations"], "0"],
        ["Citation validity", f"{_fmt_pct(o['citations']['share_valid'])} "
         f"({o['citations']['valid_refs']}/{o['citations']['total_refs']})", "100%"],
        ["Fabricated refs", o["citations"]["fabricated_count"], "0"],
        ["Planted-disqualifier recall", f"{_fmt_pct(pr['recall'])} ({pr['hits']}/"
         f"{pr['relevant']})", "100%"],
        ["Mean assessments per match", _fmt(o["coverage"]["mean_assessments"], 1), ""],
        ["Mean share unknown", _fmt_pct(o["coverage"]["mean_unknown_share"]), ""],
        ["Mean share with evidence", _fmt_pct(o["coverage"]["mean_evidence_share"]), ""],
        ["Total cost (USD)", f"{o['cost']['total_usd']:.4f}", ""],
        ["Mean wall seconds", _fmt(o["cost"]["mean_wall_seconds"], 1), ""],
        ["Runs using fallback", o["cost"]["fallback_runs"], ""],
    ]
    g = o["gold"]
    if g["gold_rows"]:
        rows.append(["Gold verdict agreement",
                     f"{_fmt_pct(g['agreement'])} ({g['agreed']}/{g['compared']})", ""])
    lines += _table(["Metric", "Value", "Target"], rows) + [""]

    lines += ["## Per patient", ""]
    prow = []
    for p in summary["patients"]:
        c, cov, cost, pl = p["citations"], p["coverage"], p["cost"], p["planted"]
        recall = "" if pl is None else (
            f"{_fmt_pct(pl['recall'])} ({pl['hits']}/{pl['relevant']})" if pl["relevant"]
            else "n/a")
        prow.append([
            p["patient_id"], p["status"], p["matches"], p["hard_constraints"]["count"],
            f"{c['valid_refs']}/{c['total_refs']}", _fmt(cov["mean_assessments"], 1),
            _fmt_pct(cov["mean_unknown_share"]), _fmt_pct(cov["mean_evidence_share"]), recall,
            cost["input_tokens"] + cost["cache_read_input_tokens"]
            + cost["cache_creation_input_tokens"], cost["output_tokens"],
            "n/a" if cost["cost_usd"] is None else f"{cost['cost_usd']:.4f}",
            _fmt(cost["wall_seconds"], 1), cost["turns"], "yes" if cost["fallback_used"] else "no",
        ])
    lines += _table(
        ["Patient", "Status", "Matches", "Hard viol.", "Valid refs", "Assess./match",
         "Unknown", "With evidence", "Planted recall", "Input tok", "Output tok", "USD",
         "Wall s", "Turns", "Fallback"], prow) + [""]

    planted = [p["planted"] for p in summary["patients"] if p["planted"] is not None]
    if planted:
        lines += ["## Planted disqualifiers", ""]
        rows = []
        for pl in planted:
            if not pl["trials"]:
                rows.append([pl["patient_id"], pl["issue"], "", "", "n/a",
                             pl.get("reason", "")])
            for t in pl["trials"]:
                rows.append([pl["patient_id"], pl["issue"], t["nct_id"], t["verdict"],
                             "hit" if t["hit"] else "miss", t["detail"]])
        lines += _table(["Patient", "Issue", "Trial", "Verdict", "Result", "Detail"], rows)
        lines += [""]

    viol = [(p["patient_id"], v) for p in summary["patients"]
            for v in p["hard_constraints"]["violations"]]
    if viol:
        lines += ["## Hard-constraint violations", ""]
        lines += _table(["Patient", "Trial", "Verdict", "Failed checks", "Found by"],
                        [[pid, v["nct_id"], v["verdict"], ", ".join(v["failed_checks"]),
                          ", ".join(v["sources"])] for pid, v in viol]) + [""]

    fab = [(p["patient_id"], f) for p in summary["patients"] for f in p["citations"]["fabricated"]]
    if fab:
        lines += ["## Fabricated evidence refs", ""]
        lines += _table(["Patient", "Trial", "Ref"],
                        [[pid, f["nct_id"], f["ref"]] for pid, f in fab]) + [""]

    errs = [(p["patient_id"], p["error"]) for p in summary["patients"] if p["error"]]
    if errs:
        lines += ["## Run errors", ""] + _table(["Patient", "Error"], errs) + [""]

    if o["cost"]["tool_calls"]:
        lines += ["## Tool calls", ""]
        lines += _table(["Tool", "Calls"], sorted(o["cost"]["tool_calls"].items())) + [""]

    if g["compared"]:
        lines += ["## Gold labels", ""]
        lines += _table(["Patient", "Trial", "Expected", "Got", "Agree"],
                        [[r["patient_id"], r["nct_id"], r["expected"], r["got"],
                          "yes" if r["agree"] else "no"] for r in g["rows"]]) + [""]

    if o["eligibility_fetch_errors"]:
        lines += ["## Unverified trials (eligibility re-fetch failed)", ""]
        lines += _table(["Trial", "Error"],
                        sorted(o["eligibility_fetch_errors"].items())) + [""]

    pricing = summary["pricing"]
    lines += [f"Prices per MTok taken {pricing['taken']} ({pricing['currency']}): "
              + "; ".join(f"{m}: input {p['input']}, output {p['output']}, cache read "
                          f"{p['cache_read']}, cache write {p['cache_write']}"
                          for m, p in pricing["per_mtok"].items()), ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        print(text.encode(enc, errors="replace").decode(enc))


def _run_one(pid: str, run: Callable[[str], MatchResult], model: str) -> MatchResult:
    try:
        return run(pid)
    except Exception as e:  # noqa: BLE001 - one bad run must not sink the eval
        return MatchResult(
            status="error",
            trace=RunTrace(run_id=uuid.uuid4().hex, patient_id=pid, model=model),
            error=f"eval harness caught {type(e).__name__}: {e}",
        )


def _out_dir(base: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = base / stamp
    n = 1
    while path.exists():
        n += 1
        path = base / f"{stamp}-{n}"
    (path / "runs").mkdir(parents=True)
    return path


def build_parser() -> argparse.ArgumentParser:

    ap = argparse.ArgumentParser(
        prog="python -m trialmatch.eval",
        description="Run trialmatch-agent over the synthetic patients and score the runs.",
    )
    ap.add_argument("--patients", default="all", help="Comma-separated ids, or 'all'.")
    ap.add_argument("--http", choices=["live", "record", "replay"], default=None,
                    help="ClinicalTrials.gov mode (default: TRIALMATCH_HTTP or live).")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--provider", choices=["anthropic", "ollama"], default="anthropic")
    ap.add_argument("--model", default=None,
                    help="Model id (default claude-opus-5-5, or qwen3:8b for ollama).")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--ollama-num-ctx", type=int, default=16_384)
    ap.add_argument("--ollama-think", action="store_true",
                    help="Turn on reasoning for local models (slower).")
    ap.add_argument("--ollama-timeout", type=float, default=1_800.0)
    ap.add_argument("--max-turns", type=int, default=25)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--expectations", default=str(DEFAULT_EXPECTATIONS))
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--no-fallback", action="store_true", help="Disable server-side fallback.")
    ap.add_argument("--yes", action="store_true", help="Skip the cost confirmation prompt.")
    return ap


def main(
    argv: list[str] | None = None,
    *,
    store: PatientStore | None = None,
    trials: TrialsSource | None = None,
    client: Any | None = None,
    input_fn: Callable[[str], str] = input,
) -> int:
    """CLI entry point. ``store``, ``trials`` and ``client`` are injectable for tests."""
    from .agent import run_match

    args = build_parser().parse_args(argv)
    if args.model is None:
        args.model = "claude-opus-5-5" if args.provider == "anthropic" else "qwen3:8b"
    paid = args.provider == "anthropic"

    if store is None:
        from .patients import FhirPatientStore

        store = FhirPatientStore()
    owned_trials = None
    if trials is None:
        from .trials import CtGovClient

        trials = owned_trials = CtGovClient(mode=args.http)

    known = store.list_ids()
    if args.patients.strip().lower() == "all":
        patient_ids = known
    else:
        patient_ids = [p.strip() for p in args.patients.split(",") if p.strip()]
        unknown = [p for p in patient_ids if p not in known]
        if unknown:
            print(f"Unknown patient ids: {', '.join(unknown)}", file=sys.stderr)
            return 2

    http_mode = args.http or getattr(trials, "mode", None) or "injected"
    print(f"About to run {len(patient_ids)} patient(s) with model {args.model} "
          f"(provider {args.provider}, effort {args.effort}, trials http mode {http_mode}). "
          + ("Each run makes paid Anthropic API calls." if paid
             else "Runs use a local model: no API cost, but they can be slow."))
    if not args.yes:
        try:
            answer = input_fn("Continue? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return 1

    try:
        if client is None and paid:
            import anthropic

            client = anthropic.Anthropic()

        def run(pid: str) -> MatchResult:
            return run_match(pid, store=store, trials=trials, client=client,
                             provider=args.provider, model=args.model,
                             effort=args.effort, max_turns=args.max_turns,
                             fallback=not args.no_fallback, ollama_url=args.ollama_url,
                             ollama_num_ctx=args.ollama_num_ctx,
                             ollama_think=args.ollama_think,
                             ollama_timeout=args.ollama_timeout)

        started = datetime.now(timezone.utc)
        if args.concurrency > 1:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                results = list(pool.map(lambda p: _run_one(p, run, args.model), patient_ids))
        else:
            results = []
            for pid in patient_ids:
                print(f"  running {pid} ...", flush=True)
                results.append(_run_one(pid, run, args.model))

        expectations = load_expectations(args.expectations)
        gold = load_gold(args.gold)
        summary = summarize(results, store, trials, expectations, gold)
    finally:
        if owned_trials is not None:
            owned_trials.close()

    out = _out_dir(Path(args.out))
    meta = {"started_utc": started.strftime("%Y-%m-%d %H:%M:%S"), "model": args.model,
            "effort": args.effort, "http": http_mode, "patients": ", ".join(patient_ids),
            "output": str(out)}
    summary["meta"] = meta
    for r in results:
        (out / "runs" / f"{r.trace.patient_id}.json").write_text(
            r.model_dump_json(indent=2), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(summary, indent=2, default=str),
                                      encoding="utf-8")
    md = render_markdown(summary, meta)
    (out / "report.md").write_text(md, encoding="utf-8")
    _safe_print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
