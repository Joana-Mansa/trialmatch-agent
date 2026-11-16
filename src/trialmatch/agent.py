"""The matching agent: a manual tool-use loop over the Claude API.

We run the loop ourselves (rather than the SDK tool runner) so we own the trace, the
turn cap, and can drive the whole thing with a scripted fake client in tests.

History is append-only: every assistant turn is appended with its full ``content``
(thinking, fallback and tool_use blocks unchanged), and all tool results for a turn
go back in one user message.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import anthropic

from .models import MatchResult, RunTrace, TraceStep
from .ports import PatientNotFound, PatientStore, TrialsSource
from .prompts import NUDGE_MESSAGE, SYSTEM_PROMPT, first_user_message
from .tools import TOOLS, ToolBox

DEFAULT_MODEL = "claude-opus-5-5"
MAX_TOKENS = 16_000
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_CONTINUATIONS = 2
CONTINUE_MESSAGE = (
    "Your last response hit the output length limit. Continue from where you stopped, "
    "keeping rationales short."
)


def build_request(
    messages: list[dict[str, Any]], *, model: str, effort: str, fallback: bool
) -> dict[str, Any]:
    """Keyword arguments for ``client.beta.messages.create``.

    The system prompt and tool list are module constants, so they are byte-stable
    across turns and runs. No ``thinking`` (always adaptive on this model), no
    sampling parameters and no ``tool_choice`` (forced choice is rejected).
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "tools": TOOLS,
        "messages": list(messages),
        "output_config": {"effort": effort},
        "cache_control": {"type": "ephemeral"},
    }
    if fallback:
        kwargs["betas"] = [FALLBACK_BETA]
        kwargs["fallbacks"] = "default"
    return kwargs


def _add_usage(trace: RunTrace, response: Any) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    u = trace.usage
    u.input_tokens += usage.input_tokens or 0
    u.output_tokens += usage.output_tokens or 0
    u.cache_read_input_tokens += usage.cache_read_input_tokens or 0
    u.cache_creation_input_tokens += usage.cache_creation_input_tokens or 0
    if any(it.type == "fallback_message" for it in (usage.iterations or [])):
        trace.fallback_used = True


def run_match(
    patient_id: str,
    *,
    store: PatientStore,
    trials: TrialsSource,
    client: Any | None = None,
    model: str = DEFAULT_MODEL,
    effort: str = "high",
    max_turns: int = 25,
    country: str | None = None,
    fallback: bool = True,
) -> MatchResult:
    started = time.perf_counter()
    trace = RunTrace(run_id=uuid.uuid4().hex, patient_id=patient_id, model=model)

    def finish(status: str, *, error: str | None = None, toolbox: ToolBox | None = None):
        report = toolbox.report if toolbox is not None else None
        issues = toolbox.validate(report) if report is not None else []
        if status == "completed" and issues:
            status = "completed_with_issues"
        trace.wall_seconds = round(time.perf_counter() - started, 3)
        return MatchResult(
            status=status, report=report, trace=trace, grounding_issues=issues, error=error
        )

    try:
        store.summary(patient_id)
    except PatientNotFound:
        return finish("error", error=f"Unknown patient id {patient_id!r}.")

    try:
        if client is None:
            client = anthropic.Anthropic()
    except anthropic.AnthropicError as e:
        return finish("error", error=f"Could not create Anthropic client: {e}")

    toolbox = ToolBox(store=store, trials=trials, patient_id=patient_id)
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": first_user_message(patient_id, country)}
    ]
    nudged = False
    continuations = 0

    while True:
        if trace.turns >= max_turns:
            return finish(
                "incomplete", error=f"Stopped after max_turns={max_turns}.", toolbox=toolbox
            )

        try:
            response = client.beta.messages.create(
                **build_request(messages, model=model, effort=effort, fallback=fallback)
            )
        except (
            anthropic.RateLimitError,
            anthropic.APIStatusError,
            anthropic.APIConnectionError,
        ) as e:
            return finish("error", error=f"{type(e).__name__}: {e}", toolbox=toolbox)
        except anthropic.AnthropicError as e:
            return finish("error", error=f"{type(e).__name__}: {e}", toolbox=toolbox)
        except TypeError as e:
            # The SDK raises a plain TypeError when no credential can be resolved.
            if "authentication" not in str(e).lower():
                raise
            return finish(
                "error",
                error="No Anthropic credential found. Set ANTHROPIC_API_KEY and retry.",
                toolbox=toolbox,
            )

        trace.turns += 1
        request_id = getattr(response, "_request_id", None)
        if request_id:
            trace.request_ids.append(request_id)
        _add_usage(trace, response)
        stop = response.stop_reason
        trace.final_stop_reason = stop

        messages.append({"role": "assistant", "content": response.content})

        if stop == "tool_use":
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                tool_input = block.input if isinstance(block.input, dict) else {}
                t0 = time.perf_counter()
                outcome = toolbox.execute(block.name, tool_input)
                trace.steps.append(
                    TraceStep(
                        step=len(trace.steps) + 1,
                        tool=block.name,
                        input=dict(tool_input),
                        ok=not outcome.is_error,
                        duration_ms=round((time.perf_counter() - t0) * 1000, 3),
                        error=outcome.content if outcome.is_error else None,
                    )
                )
                result: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": outcome.content,
                }
                if outcome.is_error:
                    result["is_error"] = True
                results.append(result)
            if results:
                messages.append({"role": "user", "content": results})
            if toolbox.report_accepted:
                return finish("completed", toolbox=toolbox)
            if not results:
                # tool_use stop without tool_use blocks should not happen; treat as no report
                return finish(
                    "incomplete", error="tool_use stop with no tool calls.", toolbox=toolbox
                )
            continue

        if stop == "pause_turn":
            continue

        if stop == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            return finish(
                "refused",
                error=f"Model refused (category: {category or 'unspecified'}).",
                toolbox=toolbox,
            )

        if stop == "max_tokens" and continuations < MAX_CONTINUATIONS:
            # Cut off by the output limit: answer any (truncated) tool calls with an
            # error so the history stays valid, then ask the model to continue.
            continuations += 1
            follow: list[dict[str, Any]] = [
                {
                    "type": "tool_result",
                    "tool_use_id": b.id,
                    "content": "This call was cut off by the output limit. Send it again, "
                    "more concisely.",
                    "is_error": True,
                }
                for b in response.content
                if b.type == "tool_use"
            ]
            follow.append({"type": "text", "text": CONTINUE_MESSAGE})
            messages.append({"role": "user", "content": follow})
            continue

        if stop in ("max_tokens", "model_context_window_exceeded"):
            return finish("incomplete", error=f"Stopped on {stop}.", toolbox=toolbox)

        # end_turn, stop_sequence or anything else: the model stopped without a report.
        if not nudged:
            nudged = True
            messages.append({"role": "user", "content": NUDGE_MESSAGE})
            continue
        return finish(
            "incomplete",
            error=f"Model ended the turn ({stop}) without submitting a report.",
            toolbox=toolbox,
        )
