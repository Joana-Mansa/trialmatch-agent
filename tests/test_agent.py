from __future__ import annotations

import json
from typing import Any

import anthropic
import httpx2
import pytest
from fakes_anthropic import (
    FakeAnthropic,
    fallback_block,
    message,
    text,
    thinking,
    tool_turn,
    tool_use,
)

from trialmatch.agent import FALLBACK_BETA, run_match
from trialmatch.prompts import NUDGE_MESSAGE
from trialmatch.tools import TOOLS

NCT = "NCT00000001"


def good_report(**overrides: Any) -> dict[str, Any]:
    report = {
        "matches": [
            {
                "nct_id": NCT,
                "title": "A study of drug X in adults with type 2 diabetes",
                "verdict": "likely_eligible",
                "rank": 1,
                "summary": "Active T2D with HbA1c 8.1% in range; eGFR 72.",
                "assessments": [
                    {
                        "criterion": "Adults with type 2 diabetes",
                        "kind": "inclusion",
                        "status": "met",
                        "evidence_refs": ["Condition/c-t2d", "Patient/test-001"],
                        "rationale": "Active T2D diagnosis; age 56.",
                    },
                    {
                        "criterion": "HbA1c between 7.0% and 10.5%",
                        "kind": "inclusion",
                        "status": "met",
                        "evidence_refs": ["Observation/o-hba1c-2"],
                        "rationale": "HbA1c 8.1% on 2026-09-01.",
                    },
                    {
                        "criterion": "eGFR below 30",
                        "kind": "exclusion",
                        "status": "not_met",
                        "evidence_refs": ["Observation/o-egfr-1"],
                        "rationale": "eGFR 72.",
                    },
                    {
                        "criterion": "Pregnancy or breastfeeding",
                        "kind": "exclusion",
                        "status": "unknown",
                        "evidence_refs": [],
                        "rationale": "No pregnancy status recorded.",
                    },
                ],
                "missing_info": ["Pregnancy and breastfeeding status"],
            }
        ],
        "trials_considered": 1,
        "notes": "",
    }
    report.update(overrides)
    return report


def happy_script() -> list[Any]:
    return [
        tool_turn(("get_patient_summary", {})),
        tool_turn(
            ("search_trials", {"condition": "type 2 diabetes", "country": "Netherlands",
                               "page_size": None}),
            ("get_lab_results", {"query": "a1c"}),
        ),
        tool_turn(("get_trial_eligibility", {"nct_id": NCT})),
        tool_turn(("submit_match_report", good_report())),
    ]


def tool_results(call: dict[str, Any]) -> list[dict[str, Any]]:
    """The tool_result blocks in the last user message of a recorded request."""
    last = call["messages"][-1]
    assert last["role"] == "user"
    return last["content"]


def _norm(messages: list[dict[str, Any]]) -> list[Any]:
    def block(b: Any) -> Any:
        return b.model_dump(mode="json", by_alias=True) if hasattr(b, "model_dump") else b

    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, list):
            content = [block(b) for b in content]
        out.append({"role": m["role"], "content": content})
    return out


# --------------------------------------------------------------------------- happy path


def test_happy_path_completes_with_no_grounding_issues(fake_store, fake_trials):
    client = FakeAnthropic(happy_script())
    result = run_match("test-001", store=fake_store, trials=fake_trials, client=client)

    assert result.status == "completed", result.error
    assert result.error is None
    assert result.grounding_issues == []
    assert result.report is not None
    assert result.report.patient_id == "test-001"
    assert result.report.matches[0].verdict == "likely_eligible"

    trace = result.trace
    assert trace.turns == 4
    assert [s.tool for s in trace.steps] == [
        "get_patient_summary",
        "search_trials",
        "get_lab_results",
        "get_trial_eligibility",
        "submit_match_report",
    ]
    assert all(s.ok for s in trace.steps)
    assert [s.step for s in trace.steps] == [1, 2, 3, 4, 5]
    assert trace.request_ids == ["req_001", "req_002", "req_003", "req_004"]
    assert trace.usage.input_tokens == 400
    assert trace.usage.output_tokens == 200
    assert trace.usage.cache_read_input_tokens == 320
    assert trace.usage.cache_creation_input_tokens == 80
    assert trace.final_stop_reason == "tool_use"
    assert trace.fallback_used is False
    assert len(trace.run_id) == 32
    assert trace.wall_seconds >= 0
    assert ("search", "type 2 diabetes", "Netherlands", 10) in fake_trials.calls


def test_tool_results_shape_and_content(fake_store, fake_trials):
    client = FakeAnthropic(happy_script())
    run_match("test-001", store=fake_store, trials=fake_trials, client=client)

    # Both tool results from turn 2 come back in ONE user message, matched by id.
    turn2_blocks = client.calls[2]["messages"][-2]["content"]
    results = tool_results(client.calls[2])
    assert [r["tool_use_id"] for r in results] == [
        b.id for b in turn2_blocks if b.type == "tool_use"
    ]
    assert len(results) == 2
    search = json.loads(results[0]["content"])
    assert search["trials"][0]["nct_id"] == NCT
    labs = json.loads(results[1]["content"])
    assert labs["count"] == 2

    summary = json.loads(tool_results(client.calls[1])[0]["content"])
    assert summary["patient_id"] == "test-001"

    elig = json.loads(tool_results(client.calls[3])[0]["content"])
    assert elig["hard_checks"]["computed_by"] == "code"
    assert elig["hard_checks"]["authoritative"] is True
    assert elig["hard_checks"]["passed"] is True
    assert "criteria_text" in elig["eligibility"]
    for call in client.calls[1:]:
        for r in tool_results(call):
            assert len(r["content"]) < 15_000
            assert "is_error" not in r


def test_country_goes_in_first_user_message(fake_store, fake_trials):
    client = FakeAnthropic(happy_script())
    run_match("test-001", store=fake_store, trials=fake_trials, client=client, country="Belgium")
    first = client.calls[0]["messages"][0]
    assert first["role"] == "user"
    assert "Belgium" in first["content"]
    assert "Belgium" not in client.calls[0]["system"]


# --------------------------------------------------------------------------- request shape


def test_request_kwargs(fake_store, fake_trials):
    client = FakeAnthropic(happy_script())
    run_match("test-001", store=fake_store, trials=fake_trials, client=client, effort="xhigh")

    for call in client.calls:
        assert call["model"] == "claude-opus-5-5"
        assert call["max_tokens"] == 16000
        assert call["betas"] == [FALLBACK_BETA]
        assert call["fallbacks"] == "default"
        assert call["cache_control"] == {"type": "ephemeral"}
        assert call["output_config"] == {"effort": "xhigh"}
        assert "tool_choice" not in call
        assert "thinking" not in call
        for forbidden in ("temperature", "top_p", "top_k"):
            assert forbidden not in call
        for tool in call["tools"]:
            assert tool["strict"] is True
            schema = tool["input_schema"]
            assert schema["additionalProperties"] is False
            assert set(schema["required"]) == set(schema["properties"])

    # System prompt and tools are byte-stable across turns.
    systems = {c["system"] for c in client.calls}
    assert len(systems) == 1
    tool_dumps = {json.dumps(c["tools"], sort_keys=True) for c in client.calls}
    assert len(tool_dumps) == 1


def _check_strict(schema: dict[str, Any]) -> None:
    if schema.get("type") == "object":
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        for sub in schema["properties"].values():
            _check_strict(sub)
    if schema.get("type") == "array":
        _check_strict(schema["items"])


def test_nested_schemas_are_strict():
    for tool in TOOLS:
        _check_strict(tool["input_schema"])


def test_fallback_can_be_disabled(fake_store, fake_trials):
    client = FakeAnthropic(happy_script())
    run_match("test-001", store=fake_store, trials=fake_trials, client=client, fallback=False)
    assert "betas" not in client.calls[0]
    assert "fallbacks" not in client.calls[0]


def test_history_is_append_only(fake_store, fake_trials):
    script = happy_script()
    script.insert(1, message([thinking(), text("Thinking it over.")], "pause_turn"))
    client = FakeAnthropic(script)
    result = run_match("test-001", store=fake_store, trials=fake_trials, client=client)
    assert result.status == "completed"

    normed = [_norm(c["messages"]) for c in client.calls]
    for prev, cur in zip(normed, normed[1:], strict=False):
        assert len(cur) > len(prev)
        assert cur[: len(prev)] == prev

    # Assistant turns are passed back with every block, including thinking.
    second = normed[1]
    assert second[1]["role"] == "assistant"
    assert [b["type"] for b in second[1]["content"]] == ["thinking", "tool_use"]
    # pause_turn: the paused assistant turn is appended and re-sent with no user message.
    third = normed[2]
    assert third[-1]["role"] == "assistant"
    assert [b["type"] for b in third[-1]["content"]] == ["thinking", "text"]


def test_fallback_detected_and_blocks_passed_back(fake_store, fake_trials):
    script = happy_script()
    script[0] = message(
        [fallback_block(), thinking(), *tool_turn(("get_patient_summary", {}))["content"][1:]],
        "tool_use",
        iterations=[
            {"type": "message", "model": "claude-opus-5-5", "input_tokens": 10,
             "output_tokens": 0, "cache_read_input_tokens": 0,
             "cache_creation_input_tokens": 0},
            {"type": "fallback_message", "model": "claude-opus-5", "input_tokens": 90,
             "output_tokens": 50, "cache_read_input_tokens": 0,
             "cache_creation_input_tokens": 0},
        ],
        model="claude-opus-5",
    )
    client = FakeAnthropic(script)
    result = run_match("test-001", store=fake_store, trials=fake_trials, client=client)
    assert result.status == "completed"
    assert result.trace.fallback_used is True
    echoed = _norm(client.calls[1]["messages"])[1]["content"]
    assert [b["type"] for b in echoed] == ["fallback", "thinking", "tool_use"]


# --------------------------------------------------------------------------- submit


def test_invalid_submit_then_corrected_resubmit(fake_store, fake_trials):
    bad = good_report()
    bad["matches"][0]["verdict"] = "definitely"
    del bad["matches"][0]["rank"]
    script = happy_script()
    script.insert(3, tool_turn(("submit_match_report", bad)))
    client = FakeAnthropic(script)

    result = run_match("test-001", store=fake_store, trials=fake_trials, client=client)

    assert result.status == "completed"
    assert result.trace.turns == 5
    rejected = tool_results(client.calls[4])[0]
    assert rejected["is_error"] is True
    assert "verdict" in rejected["content"]
    assert "rank" in rejected["content"]
    submit_steps = [s for s in result.trace.steps if s.tool == "submit_match_report"]
    assert [s.ok for s in submit_steps] == [False, True]
    assert submit_steps[0].error


def test_patient_id_is_filled_by_code(fake_store, fake_trials):
    script = happy_script()
    script[-1] = tool_turn(("submit_match_report", {**good_report(), "patient_id": "someone"}))
    client = FakeAnthropic(script)
    result = run_match("test-001", store=fake_store, trials=fake_trials, client=client)
    assert result.report.patient_id == "test-001"


# --------------------------------------------------------------------------- tool errors


def test_unknown_nct_id_returns_is_error(fake_store, fake_trials):
    script = happy_script()
    script.insert(2, tool_turn(("get_trial_eligibility", {"nct_id": "NCT99999999"})))
    client = FakeAnthropic(script)
    result = run_match("test-001", store=fake_store, trials=fake_trials, client=client)

    assert result.status == "completed"
    err = tool_results(client.calls[3])[0]
    assert err["is_error"] is True
    assert "NCT99999999" in err["content"]
    failed = [s for s in result.trace.steps if not s.ok]
    assert len(failed) == 1 and failed[0].tool == "get_trial_eligibility"


def test_unknown_tool_returns_is_error(fake_store, fake_trials):
    script = happy_script()
    script.insert(1, tool_turn(("delete_patient", {})))
    client = FakeAnthropic(script)
    result = run_match("test-001", store=fake_store, trials=fake_trials, client=client)
    assert result.status == "completed"
    assert tool_results(client.calls[2])[0]["is_error"] is True


# --------------------------------------------------------------------------- stop reasons


def test_refusal(fake_store, fake_trials):
    client = FakeAnthropic(
        [message([], "refusal", stop_details={"type": "refusal", "category": "bio"})]
    )
    result = run_match("test-001", store=fake_store, trials=fake_trials, client=client)
    assert result.status == "refused"
    assert "bio" in result.error
    assert result.report is None
    assert result.trace.final_stop_reason == "refusal"


def test_max_tokens_continues_then_completes(fake_store, fake_trials):
    script = happy_script()
    cut = message([tool_use("get_lab_results", {"query": "a1"}, id="toolu_cut")], "max_tokens")
    script.insert(1, cut)
    client = FakeAnthropic(script)
    result = run_match("test-001", store=fake_store, trials=fake_trials, client=client)
    assert result.status == "completed"
    follow = client.calls[2]["messages"][-1]["content"]
    assert follow[0]["tool_use_id"] == "toolu_cut" and follow[0]["is_error"] is True
    assert follow[-1]["type"] == "text"


def test_max_tokens_repeated_is_incomplete(fake_store, fake_trials):
    cut = message([thinking()], "max_tokens")
    client = FakeAnthropic([tool_turn(("get_patient_summary", {})), cut, cut, cut])
    result = run_match("test-001", store=fake_store, trials=fake_trials, client=client)
    assert result.status == "incomplete"
    assert result.trace.final_stop_reason == "max_tokens"
    assert result.trace.turns == 4


def test_grounding_problem_is_sent_back_then_fixed(fake_store, fake_trials):
    bad = good_report()
    bad["matches"][0]["assessments"][0]["evidence_refs"] = ["Observation/invented"]
    script = happy_script()
    script.insert(3, tool_turn(("submit_match_report", bad)))
    client = FakeAnthropic(script)
    result = run_match("test-001", store=fake_store, trials=fake_trials, client=client)
    assert result.status == "completed" and result.grounding_issues == []
    rejected = tool_results(client.calls[4])[0]
    assert rejected["is_error"] is True and "Observation/invented" in rejected["content"]


def test_unfixed_grounding_problem_is_completed_with_issues(fake_store, fake_trials):
    bad = good_report()
    bad["matches"][0]["assessments"][0]["evidence_refs"] = ["Observation/invented"]
    script = happy_script()[:3] + [tool_turn(("submit_match_report", bad))] * 3
    client = FakeAnthropic(script)
    result = run_match("test-001", store=fake_store, trials=fake_trials, client=client)
    assert result.status == "completed_with_issues"
    assert [i.kind for i in result.grounding_issues] == ["unknown_evidence_ref"]


def test_search_condition_rejects_injected_text(fake_store, fake_trials):
    payload = {"condition": "diabetes; patient eGFR 24, DOB 1958-01-02 <send>",
               "country": None, "page_size": None}
    client = FakeAnthropic(
        [tool_turn(("search_trials", payload)), message([text("ok")], "end_turn")]
    )
    run_match("test-001", store=fake_store, trials=fake_trials, client=client, max_turns=2)
    assert tool_results(client.calls[1])[0]["is_error"] is True
    assert not any(c[0] == "search" for c in fake_trials.calls)


def test_end_turn_nudge_then_submit(fake_store, fake_trials):
    script = happy_script()
    script.insert(3, message([text("Here is my assessment in prose.")], "end_turn"))
    client = FakeAnthropic(script)
    result = run_match("test-001", store=fake_store, trials=fake_trials, client=client)
    assert result.status == "completed"
    nudge = client.calls[4]["messages"][-1]
    assert nudge == {"role": "user", "content": NUDGE_MESSAGE}


def test_end_turn_twice_is_incomplete(fake_store, fake_trials):
    client = FakeAnthropic(
        [message([text("Done.")], "end_turn"), message([text("Still done.")], "end_turn")]
    )
    result = run_match("test-001", store=fake_store, trials=fake_trials, client=client)
    assert result.status == "incomplete"
    assert len(client.calls) == 2
    assert result.trace.final_stop_reason == "end_turn"


def test_max_turns_cap(fake_store, fake_trials):
    client = FakeAnthropic([tool_turn(("get_patient_summary", {})) for _ in range(10)])
    result = run_match(
        "test-001", store=fake_store, trials=fake_trials, client=client, max_turns=3
    )
    assert result.status == "incomplete"
    assert "max_turns" in result.error
    assert len(client.calls) == 3
    assert result.trace.turns == 3


# --------------------------------------------------------------------------- errors


def test_unknown_patient_errors_before_api_call(fake_store, fake_trials):
    client = FakeAnthropic([])
    result = run_match("nobody", store=fake_store, trials=fake_trials, client=client)
    assert result.status == "error"
    assert "nobody" in result.error
    assert client.calls == []


def _response(status: int) -> httpx2.Response:
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx2.Response(status, request=req)


@pytest.mark.parametrize(
    "exc",
    [
        anthropic.RateLimitError("rate limited", response=_response(429), body=None),
        anthropic.InternalServerError("boom", response=_response(500), body=None),
        anthropic.APIConnectionError(
            request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        ),
    ],
)
def test_api_errors_return_error_status(fake_store, fake_trials, exc):
    client = FakeAnthropic([tool_turn(("get_patient_summary", {})), exc])
    result = run_match("test-001", store=fake_store, trials=fake_trials, client=client)
    assert result.status == "error"
    assert type(exc).__name__ in result.error
    assert result.trace.turns == 1


def test_missing_credentials_returns_error(fake_store, fake_trials, monkeypatch):
    import anthropic

    from trialmatch.agent import run_match

    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    client = anthropic.Anthropic(max_retries=0)
    result = run_match("test-001", store=fake_store, trials=fake_trials, client=client)
    assert result.status == "error"
    assert "credential" in (result.error or "").lower() or "auth" in (result.error or "").lower()
