"""Command line interface.

    trialmatch patients
    trialmatch match --patient p001 [--country Netherlands] [--effort high] [--json] [--out f.json]
    trialmatch serve [--host 127.0.0.1] [--port 8000]

``match`` exit codes:

* 0 completed
* 3 completed with issues (a report exists but code grounding found problems; the
  report needs human review)
* 2 refused or incomplete
* 1 error (including a missing credential or unknown patient)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

from . import config
from .models import MatchResult
from .ports import PatientNotFound, PatientStore, TrialsSource

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NOT_COMPLETED = 2
EXIT_NEEDS_REVIEW = 3

MISSING_KEY_MESSAGE = """\
No Anthropic credential found, so the agent cannot call Claude.

Set ANTHROPIC_API_KEY for this shell session and run the command again:

  PowerShell:  $env:ANTHROPIC_API_KEY = "sk-ant-..."
  bash/zsh:    export ANTHROPIC_API_KEY="sk-ant-..."

Get a key at https://platform.claude.com/settings/keys. To try the CLI without a key,
`trialmatch patients` lists the bundled synthetic patients.
"""


def _exit_code(status: str) -> int:
    if status == "completed":
        return EXIT_OK
    if status == "completed_with_issues":
        return EXIT_NEEDS_REVIEW
    if status in ("refused", "incomplete"):
        return EXIT_NOT_COMPLETED
    return EXIT_ERROR


# --------------------------------------------------------------------------- rendering


def render_report(result: MatchResult) -> str:
    """A plain-text report: matches with criterion-level evidence, then a run footer."""
    t = result.trace
    lines: list[str] = []
    if result.grounding_issues or result.status == "completed_with_issues":
        banner = (
            f"NEEDS REVIEW: grounding issues ({len(result.grounding_issues)}); "
            "check them before relying on this report"
        )
        lines += ["!" * 72, banner, "!" * 72]
    lines.append(f"Patient {t.patient_id}  status: {result.status}")
    if result.error:
        lines.append(f"Error: {result.error}")
    report = result.report
    if report is not None:
        lines.append(f"Trials considered: {report.trials_considered}")
        if not report.matches:
            lines.append("No matching trials reported.")
        for m in sorted(report.matches, key=lambda m: m.rank):
            lines += ["", f"#{m.rank}  {m.nct_id}  {m.verdict}  {m.title}", f"    {m.summary}"]
            for a in m.assessments:
                refs = f"  <- {', '.join(a.evidence_refs)}" if a.evidence_refs else ""
                lines.append(f"    [{a.status}] ({a.kind}) {a.criterion}{refs}")
            if m.missing_info:
                lines.append("    Missing info:")
                lines += [f"      - {item}" for item in m.missing_info]
        if report.notes:
            lines += ["", f"Notes: {report.notes}"]
    if result.grounding_issues:
        lines += ["", "Grounding issues:"]
        for g in result.grounding_issues:
            where = f" {g.nct_id}" if g.nct_id else ""
            lines.append(f"  - {g.kind}{where}: {g.detail}")
    u = t.usage
    footer = (
        f"turns {t.turns} | tool calls {len(t.steps)} | tokens in {u.input_tokens} "
        f"out {u.output_tokens} (cache read {u.cache_read_input_tokens}, "
        f"write {u.cache_creation_input_tokens}) | wall {t.wall_seconds:.1f}s | model {t.model}"
    )
    if t.fallback_used:
        footer += " | fallback used"
    lines += ["", "-" * 72, footer]
    return "\n".join(lines)


# --------------------------------------------------------------------------- commands


def _cmd_patients(store: PatientStore, out: TextIO) -> int:
    from .api import patient_listing

    for pid in store.list_ids():
        p = patient_listing(store.summary(pid))
        print(
            f"{p.patient_id:<10} {p.sex:<7} {p.age_years:>3}y  {p.primary_condition or '-'}",
            file=out,
        )
    return EXIT_OK


def _cmd_match(
    args: argparse.Namespace,
    settings: config.Settings,
    store: PatientStore,
    trials: TrialsSource | None,
    run_fn: Callable[..., MatchResult] | None,
    out: TextIO,
    err: TextIO,
) -> int:
    try:
        store.summary(args.patient)
    except PatientNotFound:
        known = ", ".join(store.list_ids()) or "none"
        print(f"Unknown patient id {args.patient!r}. Known ids: {known}", file=err)
        return EXIT_ERROR

    if args.provider or args.model:
        from dataclasses import replace

        provider = args.provider or settings.provider
        model = args.model or (
            config.DEFAULT_MODELS[provider] if args.provider else settings.model
        )
        settings = replace(settings, provider=provider, model=model)

    if run_fn is None:
        if settings.provider == "anthropic" and not config.credentials_available():
            print(MISSING_KEY_MESSAGE, file=err)
            return EXIT_ERROR
        from .agent import run_match

        run_fn = run_match

    result = run_fn(
        args.patient,
        store=store,
        trials=trials if trials is not None else config.make_trials(settings),
        **settings.run_kwargs(),
        effort=args.effort or settings.effort,
        max_turns=settings.max_turns,
        country=args.country,
    )

    payload = result.model_dump_json(indent=2)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    if args.json:
        print(payload, file=out)
    else:
        print(render_report(result), file=out)
        if args.out:
            print(f"Saved JSON to {args.out}", file=out)
    return _exit_code(result.status)


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trialmatch",
        description="Match synthetic FHIR patients to recruiting ClinicalTrials.gov trials.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("patients", help="List patients with age, sex and primary condition.")

    m = sub.add_parser("match", help="Run the matching agent for one patient.")
    m.add_argument("--patient", required=True, help="Patient id, e.g. p001.")
    m.add_argument("--country", default=None, help="Restrict trial search to a country.")
    m.add_argument(
        "--effort",
        choices=config.EFFORTS,
        default=None,
        help="Model effort (default: TRIALMATCH_EFFORT or high).",
    )
    m.add_argument(
        "--provider", choices=list(config.PROVIDERS), default=None,
        help="Model provider: anthropic (Claude, default) or ollama (local model).",
    )
    m.add_argument("--model", default=None, help="Model id, e.g. claude-opus-5-5 or qwen3:8b.")
    m.add_argument("--json", action="store_true", help="Print the MatchResult as JSON.")
    m.add_argument("--out", default=None, help="Also write the MatchResult JSON to this file.")

    s = sub.add_parser("serve", help="Run the HTTP API.")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    run_fn: Callable[..., MatchResult] | None = None,
    store: PatientStore | None = None,
    trials: TrialsSource | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Entry point. ``run_fn``, ``store`` and ``trials`` are test seams."""
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    for stream in (out, err):  # criteria text is often non-ASCII; never crash a cp1252 console
        reconfigure: Any = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except Exception:
                pass

    args = build_parser().parse_args(argv)
    if args.command == "serve":
        return _cmd_serve(args)

    try:
        settings = config.get_settings()
        store = store if store is not None else config.make_store(settings)
    except (FileNotFoundError, ValueError) as e:
        print(f"Configuration error: {e}", file=err)
        return EXIT_ERROR

    if args.command == "patients":
        return _cmd_patients(store, out)
    return _cmd_match(args, settings, store, trials, run_fn, out, err)


if __name__ == "__main__":
    sys.exit(main())
