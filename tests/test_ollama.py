"""The local-model provider, driven by a scripted fake Ollama server (no Ollama needed)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from test_agent import NCT, good_report

from trialmatch import config
from trialmatch import eval as ev
from trialmatch.agent import LOCAL_TRIAL_BUDGET, run_match
from trialmatch.llm import ModelError, OllamaChat, ToolResult, make_chat_model
from trialmatch.models import Usage
from trialmatch.tools import TOOLS


def reply(*calls: tuple[str, Any], content: str = "", done_reason: str = "stop",
          with_ids: bool = False) -> dict[str, Any]:
    tool_calls = []
    for i, (name, args) in enumerate(calls):
        call: dict[str, Any] = {"function": {"name": name, "arguments": args}}
        if with_ids:
            call["id"] = f"call_{i}"
        tool_calls.append(call)
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"message": msg, "done": True, "done_reason": done_reason,
            "prompt_eval_count": 1000, "eval_count": 200}


class FakeOllama:
    """Returns scripted /api/chat replies and records each request body."""

    def __init__(self, script: list[Any]):
        self.script = list(script)
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        self.bodies.append(json.loads(request.content))
        item = self.script.pop(0)
        if isinstance(item, httpx.Response):
            return item
        if isinstance(item, Exception):
            raise item
        return httpx.Response(200, json=item)

    def chat(self, **kw: Any) -> OllamaChat:
        return OllamaChat(http=httpx.Client(transport=httpx.MockTransport(self)), **kw)


def happy_ollama_script() -> list[dict[str, Any]]:
    return [
        reply(("get_patient_summary", {})),
        reply(("search_trials", {"condition": "type 2 diabetes", "country": None,
                                 "page_size": None})),
        # Some models return arguments as a JSON string; the adapter parses it.
        reply(("get_trial_eligibility", json.dumps({"nct_id": NCT}))),
        reply(("submit_match_report", good_report()), with_ids=True),
    ]


def test_happy_path_through_ollama(fake_store, fake_trials):
    server = FakeOllama(happy_ollama_script())
    result = run_match("test-001", store=fake_store, trials=fake_trials,
                       llm=server.chat(model="qwen3:8b"), provider="ollama")
    assert result.status == "completed", result.error
    assert result.grounding_issues == []
    assert result.trace.model == "ollama:qwen3:8b"
    assert [s.tool for s in result.trace.steps] == [
        "get_patient_summary", "search_trials", "get_trial_eligibility", "submit_match_report"
    ]
    assert result.trace.usage.input_tokens == 4000 and result.trace.usage.output_tokens == 800

    first = server.bodies[0]
    assert first["model"] == "qwen3:8b" and first["stream"] is False
    assert first["think"] is False  # reasoning off by default for speed
    assert first["options"]["num_ctx"] >= 16_384  # Ollama's default is far too small
    assert first["messages"][0]["role"] == "system"
    assert f"at most {LOCAL_TRIAL_BUDGET} trials" in first["messages"][1]["content"]
    assert {t["function"]["name"] for t in first["tools"]} == {t["name"] for t in TOOLS}
    assert all(t["type"] == "function" for t in first["tools"])

    # Tool results go back as role "tool" messages naming the tool.
    last = server.bodies[-1]["messages"]
    assert last[-1]["role"] == "tool" and last[-1]["tool_name"] == "get_trial_eligibility"
    assert last[-2]["role"] == "assistant" and last[-2]["tool_calls"]


def test_tool_errors_are_marked_for_the_model(fake_store, fake_trials):
    server = FakeOllama([
        reply(("get_trial_eligibility", {"nct_id": "NCT99999999"})),
        reply(content="I could not find it.", done_reason="stop"),
        reply(content="Still nothing.", done_reason="stop"),
    ])
    result = run_match("test-001", store=fake_store, trials=fake_trials,
                       llm=server.chat(), provider="ollama")
    assert result.status == "incomplete"
    tool_msg = server.bodies[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["content"].startswith("ERROR:")


def test_length_stop_continues(fake_store, fake_trials):
    script = happy_ollama_script()
    script.insert(1, reply(content="partial", done_reason="length"))
    server = FakeOllama(script)
    result = run_match("test-001", store=fake_store, trials=fake_trials,
                       llm=server.chat(), provider="ollama")
    assert result.status == "completed"
    assert server.bodies[2]["messages"][-1]["role"] == "user"


def test_missing_model_and_unreachable_server_are_clear_errors(fake_store, fake_trials):
    server = FakeOllama([httpx.Response(404, json={"error": "model not found"})])
    result = run_match("test-001", store=fake_store, trials=fake_trials,
                       llm=server.chat(model="qwen3:8b"), provider="ollama")
    assert result.status == "error" and "ollama pull qwen3:8b" in result.error

    server = FakeOllama([httpx.ConnectError("refused")])
    result = run_match("test-001", store=fake_store, trials=fake_trials,
                       llm=server.chat(), provider="ollama")
    assert result.status == "error" and "ollama serve" in result.error


def test_unparseable_arguments_reach_the_tool_as_an_error(fake_store, fake_trials):
    chat = FakeOllama([reply(("get_lab_results", "{not json"))]).chat()
    chat.start("sys", TOOLS, "hi")
    turn = chat.next_turn()
    assert turn.stop == "tool_use"
    assert turn.tool_calls[0].input == {"_unparsed_arguments": "{not json"}
    assert turn.tool_calls[0].id.startswith("call_")


def test_add_tool_results_format():
    chat = OllamaChat(http=httpx.Client(transport=httpx.MockTransport(lambda r: None)))
    chat.start("sys", TOOLS, "hi")
    chat.add_tool_results([ToolResult("c1", "search_trials", "boom", is_error=True)],
                          text="continue")
    assert chat.messages[-2] == {"role": "tool", "content": "ERROR: boom",
                                 "tool_name": "search_trials"}
    assert chat.messages[-1] == {"role": "user", "content": "continue"}


def test_claude_default_model_is_not_sent_to_ollama(fake_store, fake_trials, monkeypatch):
    captured = {}

    def fake_make(provider, **kw):
        captured.update(kw, provider=provider)
        raise ModelError("stop here")

    monkeypatch.setattr("trialmatch.agent.make_chat_model", fake_make)
    result = run_match("test-001", store=fake_store, trials=fake_trials, provider="ollama",
                       model="claude-opus-5-5")
    assert captured["provider"] == "ollama" and captured["model"] is None
    assert result.status == "error"


def test_make_chat_model_rejects_unknown_provider():
    with pytest.raises(ValueError):
        make_chat_model("openai")


def test_settings_provider_defaults():
    s = config.Settings.from_env({"TRIALMATCH_PROVIDER": "ollama"})
    assert s.provider == "ollama" and s.model == "qwen3:8b"
    assert s.run_kwargs()["ollama_url"] == "http://localhost:11434"
    assert config.Settings.from_env({}).model == "claude-opus-5-5"
    with pytest.raises(ValueError):
        config.Settings.from_env({"TRIALMATCH_PROVIDER": "gpt"})


def test_local_runs_cost_nothing():
    usage = Usage(input_tokens=50_000, output_tokens=9_000)
    assert ev.usage_cost(usage, "ollama:qwen3:8b") == 0.0


def test_timeout_message_is_not_a_connection_error(fake_store, fake_trials):
    server = FakeOllama([httpx.ReadTimeout("slow")])
    result = run_match("test-001", store=fake_store, trials=fake_trials,
                       llm=server.chat(timeout=5), provider="ollama")
    assert result.status == "error"
    assert "did not answer within 5s" in result.error and "ollama serve" not in result.error


def test_think_and_timeout_settings():
    s = config.Settings.from_env({"TRIALMATCH_PROVIDER": "ollama",
                                  "TRIALMATCH_OLLAMA_THINK": "1",
                                  "TRIALMATCH_OLLAMA_TIMEOUT": "60"})
    kw = s.run_kwargs()
    assert kw["ollama_think"] is True and kw["ollama_timeout"] == 60.0
    assert kw["ollama_num_ctx"] == 16_384
