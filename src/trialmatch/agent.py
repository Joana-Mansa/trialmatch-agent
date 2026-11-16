"""The matching agent: a manual tool-use loop over a ``ChatModel`` provider.

We run the loop ourselves (rather than an SDK tool runner) so we own the trace, the
turn cap, and can drive the whole thing with scripted fakes in tests. The loop is
provider-neutral: Claude (reference) or a local Ollama model plug in through
``llm.ChatModel``, and both are held to the same code checks.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from .llm import (
    ANTHROPIC_DEFAULT_MODEL,
    FALLBACK_BETA,
    OLLAMA_DEFAULT_NUM_CTX,
    OLLAMA_DEFAULT_URL,
    ChatModel,
    ModelError,
    ToolResult,
    build_anthropic_request,
    make_chat_model,
)
from .models import MatchResult, RunTrace, TraceStep
from .ports import PatientNotFound, PatientStore, TrialsSource
from .prompts import NUDGE_MESSAGE, SYSTEM_PROMPT, first_user_message
from .tools import TOOLS, ToolBox

DEFAULT_MODEL = ANTHROPIC_DEFAULT_MODEL
MAX_CONTINUATIONS = 2
LOCAL_TRIAL_BUDGET = 4  # small local models: assess fewer trials to fit their context
CONTINUE_MESSAGE = (
    "Your last response hit the output length limit. Continue from where you stopped, "
    "keeping rationales short."
)

__all__ = ["DEFAULT_MODEL", "FALLBACK_BETA", "build_request", "run_match"]


def build_request(
    messages: list[dict[str, Any]], *, model: str, effort: str, fallback: bool
) -> dict[str, Any]:
    """Claude request kwargs for this agent's system prompt and tools."""
    return build_anthropic_request(
        messages, system=SYSTEM_PROMPT, tools=TOOLS, model=model, effort=effort,
        fallback=fallback,
    )


def run_match(
    patient_id: str,
    *,
    store: PatientStore,
    trials: TrialsSource,
    client: Any | None = None,
    llm: ChatModel | None = None,
    provider: str = "anthropic",
    model: str | None = DEFAULT_MODEL,
    effort: str = "high",
    max_turns: int = 25,
    country: str | None = None,
    fallback: bool = True,
    trial_budget: int | None = None,
    ollama_url: str = OLLAMA_DEFAULT_URL,
    ollama_num_ctx: int = OLLAMA_DEFAULT_NUM_CTX,
) -> MatchResult:
    started = time.perf_counter()
    if provider == "ollama" and model and model.startswith("claude"):
        model = None  # a Claude default leaking in from config; use the local default
    trace = RunTrace(
        run_id=uuid.uuid4().hex,
        patient_id=patient_id,
        model=model or provider,
    )

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

    if llm is None:
        try:
            llm = make_chat_model(
                provider, model=model, effort=effort, fallback=fallback, client=client,
                ollama_url=ollama_url, ollama_num_ctx=ollama_num_ctx,
            )
        except ValueError as e:
            return finish("error", error=str(e))
        except Exception as e:  # noqa: BLE001 - e.g. SDK credential errors at construction
            return finish("error", error=f"Could not create {provider} client: {e}")
    trace.model = llm.name

    if trial_budget is None and provider == "ollama":
        trial_budget = LOCAL_TRIAL_BUDGET

    toolbox = ToolBox(store=store, trials=trials, patient_id=patient_id)
    llm.start(SYSTEM_PROMPT, TOOLS, first_user_message(patient_id, country, trial_budget))
    nudged = False
    continuations = 0

    while True:
        if trace.turns >= max_turns:
            return finish(
                "incomplete", error=f"Stopped after max_turns={max_turns}.", toolbox=toolbox
            )

        try:
            turn = llm.next_turn()
        except ModelError as e:
            return finish("error", error=str(e), toolbox=toolbox)

        trace.turns += 1
        if turn.request_id:
            trace.request_ids.append(turn.request_id)
        u = trace.usage
        u.input_tokens += turn.usage.input_tokens
        u.output_tokens += turn.usage.output_tokens
        u.cache_read_input_tokens += turn.usage.cache_read_input_tokens
        u.cache_creation_input_tokens += turn.usage.cache_creation_input_tokens
        trace.fallback_used = trace.fallback_used or turn.fallback_used
        trace.final_stop_reason = turn.raw_stop

        if turn.stop == "tool_use":
            results: list[ToolResult] = []
            for call in turn.tool_calls:
                t0 = time.perf_counter()
                outcome = toolbox.execute(call.name, call.input)
                trace.steps.append(
                    TraceStep(
                        step=len(trace.steps) + 1,
                        tool=call.name,
                        input=dict(call.input),
                        ok=not outcome.is_error,
                        duration_ms=round((time.perf_counter() - t0) * 1000, 3),
                        error=outcome.content if outcome.is_error else None,
                    )
                )
                results.append(
                    ToolResult(call.id, call.name, outcome.content, outcome.is_error)
                )
            llm.add_tool_results(results)
            if toolbox.report_accepted:
                return finish("completed", toolbox=toolbox)
            if not results:
                return finish(
                    "incomplete", error="tool_use stop with no tool calls.", toolbox=toolbox
                )
            continue

        if turn.stop == "pause":
            continue

        if turn.stop == "refusal":
            return finish(
                "refused",
                error=f"Model refused (category: {turn.detail or 'unspecified'}).",
                toolbox=toolbox,
            )

        if turn.stop == "max_tokens" and continuations < MAX_CONTINUATIONS:
            # Cut off by the output limit: answer any (truncated) tool calls with an
            # error so the history stays valid, then ask the model to continue.
            continuations += 1
            cut = [
                ToolResult(
                    c.id, c.name,
                    "This call was cut off by the output limit. Send it again, more concisely.",
                    is_error=True,
                )
                for c in turn.tool_calls
            ]
            llm.add_tool_results(cut, text=CONTINUE_MESSAGE)
            continue

        if turn.stop in ("max_tokens", "context_exceeded"):
            return finish("incomplete", error=f"Stopped on {turn.raw_stop}.", toolbox=toolbox)

        # end_turn or anything else: the model stopped without a report.
        if not nudged:
            nudged = True
            llm.add_user_text(NUDGE_MESSAGE)
            continue
        return finish(
            "incomplete",
            error=f"Model ended the turn ({turn.raw_stop}) without submitting a report.",
            toolbox=toolbox,
        )
