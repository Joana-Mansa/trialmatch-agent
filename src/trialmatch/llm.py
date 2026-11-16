"""Model providers behind one small interface.

The agent loop only needs four things from a model: start a conversation, take the
next turn, accept tool results, accept a user message. Each provider keeps its own
history in its native format, so provider rules (for Claude: append the full content
of every turn unchanged) stay inside the adapter.

* ``AnthropicChat``: Claude through the Messages API (the reference provider).
* ``OllamaChat``: a local open-weights model through Ollama's ``/api/chat``. Free and
  on-premises: no patient data leaves the machine, which is what many hospitals require.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx

from .models import Usage

StopKind = Literal["tool_use", "end_turn", "max_tokens", "refusal", "pause", "context_exceeded"]


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class ModelTurn:
    stop: StopKind
    raw_stop: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    text: str = ""
    usage: Usage = field(default_factory=Usage)
    request_id: str | None = None
    fallback_used: bool = False
    detail: str | None = None  # e.g. refusal category


class ModelError(RuntimeError):
    """The provider could not produce a turn (auth, network, bad request, server down)."""


class ChatModel(Protocol):
    name: str  # model id recorded in the trace

    def start(self, system: str, tools: list[dict[str, Any]], first_user: str) -> None: ...

    def next_turn(self) -> ModelTurn: ...

    def add_tool_results(self, results: list[ToolResult], text: str | None = None) -> None: ...

    def add_user_text(self, text: str) -> None: ...


# --------------------------------------------------------------------------- Claude

ANTHROPIC_DEFAULT_MODEL = "claude-opus-5-5"
ANTHROPIC_MAX_TOKENS = 16_000
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def build_anthropic_request(
    messages: list[dict[str, Any]],
    *,
    system: str,
    tools: list[dict[str, Any]],
    model: str,
    effort: str,
    fallback: bool,
) -> dict[str, Any]:
    """Keyword arguments for ``client.beta.messages.create``.

    The system prompt and tool list are constants, so they are byte-stable across turns
    and prompt caching hits. No ``thinking`` (always adaptive on this model), no
    sampling parameters and no ``tool_choice`` (forced choice is rejected).
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": ANTHROPIC_MAX_TOKENS,
        "system": system,
        "tools": tools,
        "messages": list(messages),
        "output_config": {"effort": effort},
        "cache_control": {"type": "ephemeral"},
    }
    if fallback:
        kwargs["betas"] = [FALLBACK_BETA]
        kwargs["fallbacks"] = "default"
    return kwargs


class AnthropicChat:
    """Claude via ``client.beta.messages.create``. History is append-only."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        model: str = ANTHROPIC_DEFAULT_MODEL,
        effort: str = "high",
        fallback: bool = True,
    ) -> None:
        import anthropic

        self._anthropic = anthropic
        self.client = client if client is not None else anthropic.Anthropic()
        self.name = model
        self.effort = effort
        self.fallback = fallback
        self.messages: list[dict[str, Any]] = []
        self._system = ""
        self._tools: list[dict[str, Any]] = []

    def start(self, system: str, tools: list[dict[str, Any]], first_user: str) -> None:
        self._system, self._tools = system, tools
        self.messages = [{"role": "user", "content": first_user}]

    def next_turn(self) -> ModelTurn:
        a = self._anthropic
        try:
            response = self.client.beta.messages.create(
                **build_anthropic_request(
                    self.messages,
                    system=self._system,
                    tools=self._tools,
                    model=self.name,
                    effort=self.effort,
                    fallback=self.fallback,
                )
            )
        except a.AnthropicError as e:  # API status, rate limit, connection, credentials
            raise ModelError(f"{type(e).__name__}: {e}") from e
        except TypeError as e:
            # The SDK raises a plain TypeError when no credential can be resolved.
            if "authentication" not in str(e).lower():
                raise
            raise ModelError(
                "No Anthropic credential found. Set ANTHROPIC_API_KEY and retry."
            ) from e

        # Append the full content unchanged (thinking, fallback and tool_use blocks).
        self.messages.append({"role": "assistant", "content": response.content})

        usage = Usage()
        fallback_used = False
        u = getattr(response, "usage", None)
        if u is not None:
            usage = Usage(
                input_tokens=u.input_tokens or 0,
                output_tokens=u.output_tokens or 0,
                cache_read_input_tokens=u.cache_read_input_tokens or 0,
                cache_creation_input_tokens=u.cache_creation_input_tokens or 0,
            )
            fallback_used = any(
                it.type == "fallback_message" for it in (u.iterations or [])
            )

        raw = response.stop_reason or ""
        stop: StopKind = {
            "tool_use": "tool_use",
            "pause_turn": "pause",
            "refusal": "refusal",
            "max_tokens": "max_tokens",
            "model_context_window_exceeded": "context_exceeded",
        }.get(raw, "end_turn")  # type: ignore[assignment]
        detail = None
        if stop == "refusal":
            details = getattr(response, "stop_details", None)
            detail = getattr(details, "category", None) if details else None

        calls = [
            ToolCall(id=b.id, name=b.name, input=b.input if isinstance(b.input, dict) else {})
            for b in response.content
            if b.type == "tool_use"
        ]
        text = "".join(b.text for b in response.content if b.type == "text")
        return ModelTurn(
            stop=stop,
            raw_stop=raw,
            tool_calls=calls,
            text=text,
            usage=usage,
            request_id=getattr(response, "_request_id", None),
            fallback_used=fallback_used,
            detail=detail,
        )

    def add_tool_results(self, results: list[ToolResult], text: str | None = None) -> None:
        # All tool results for one assistant turn go back in a single user message.
        content: list[dict[str, Any]] = []
        for r in results:
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": r.tool_call_id,
                "content": r.content,
            }
            if r.is_error:
                block["is_error"] = True
            content.append(block)
        if text:
            content.append({"type": "text", "text": text})
        if content:
            self.messages.append({"role": "user", "content": content})

    def add_user_text(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})


# --------------------------------------------------------------------------- Ollama

OLLAMA_DEFAULT_MODEL = "qwen3:8b"
OLLAMA_DEFAULT_URL = "http://localhost:11434"
OLLAMA_DEFAULT_NUM_CTX = 24_576  # Ollama's own default (a few thousand) is far too small


def _ollama_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool["input_schema"],
        },
    }


class OllamaChat:
    """A local model via Ollama's native ``/api/chat`` (supports tools and ``num_ctx``).

    Small local models are weaker at long tool-use runs than Claude. The code checks
    downstream (hard checks, grounding) apply to every provider, which is the point:
    they catch what a weaker model gets wrong.
    """

    def __init__(
        self,
        *,
        model: str = OLLAMA_DEFAULT_MODEL,
        base_url: str = OLLAMA_DEFAULT_URL,
        num_ctx: int = OLLAMA_DEFAULT_NUM_CTX,
        num_predict: int = 8_192,
        timeout: float = 600.0,
        http: httpx.Client | None = None,
    ) -> None:
        self.name = f"ollama:{model}"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.http = http or httpx.Client(timeout=timeout)
        self.messages: list[dict[str, Any]] = []
        self._tools: list[dict[str, Any]] = []

    def start(self, system: str, tools: list[dict[str, Any]], first_user: str) -> None:
        self._tools = [_ollama_tool(t) for t in tools]
        self.messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": first_user},
        ]

    def next_turn(self) -> ModelTurn:
        body = {
            "model": self.model,
            "messages": self.messages,
            "tools": self._tools,
            "stream": False,
            "options": {"num_ctx": self.num_ctx, "num_predict": self.num_predict},
        }
        try:
            resp = self.http.post(f"{self.base_url}/api/chat", json=body)
        except httpx.HTTPError as e:
            raise ModelError(
                f"Could not reach Ollama at {self.base_url} ({type(e).__name__}). Is it "
                "running? Start it with `ollama serve` or the Ollama app."
            ) from e
        if resp.status_code == 404:
            raise ModelError(
                f"Ollama has no model {self.model!r}. Pull it with `ollama pull {self.model}`."
            )
        if resp.status_code >= 400:
            raise ModelError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        msg = data.get("message") or {}
        calls: list[ToolCall] = []
        for raw_call in msg.get("tool_calls") or []:
            fn = raw_call.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_unparsed_arguments": args}
            calls.append(
                ToolCall(
                    id=raw_call.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                    name=fn.get("name", ""),
                    input=args if isinstance(args, dict) else {},
                )
            )

        # Keep the assistant message as returned so the model sees its own calls.
        assistant: dict[str, Any] = {"role": "assistant", "content": msg.get("content", "")}
        if msg.get("tool_calls"):
            assistant["tool_calls"] = msg["tool_calls"]
        self.messages.append(assistant)

        raw = data.get("done_reason") or ""
        if calls:
            stop: StopKind = "tool_use"
        elif raw == "length":
            stop = "max_tokens"
        else:
            stop = "end_turn"
        usage = Usage(
            input_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
        )
        return ModelTurn(
            stop=stop,
            raw_stop=raw or ("tool_calls" if calls else "stop"),
            tool_calls=calls,
            text=msg.get("content", "") or "",
            usage=usage,
        )

    def add_tool_results(self, results: list[ToolResult], text: str | None = None) -> None:
        for r in results:
            content = f"ERROR: {r.content}" if r.is_error else r.content
            self.messages.append({"role": "tool", "content": content, "tool_name": r.name})
        if text:
            self.messages.append({"role": "user", "content": text})

    def add_user_text(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})


def make_chat_model(
    provider: str,
    *,
    model: str | None = None,
    effort: str = "high",
    fallback: bool = True,
    client: Any | None = None,
    ollama_url: str = OLLAMA_DEFAULT_URL,
    ollama_num_ctx: int = OLLAMA_DEFAULT_NUM_CTX,
) -> ChatModel:
    if provider == "anthropic":
        return AnthropicChat(
            client, model=model or ANTHROPIC_DEFAULT_MODEL, effort=effort, fallback=fallback
        )
    if provider == "ollama":
        return OllamaChat(
            model=model or OLLAMA_DEFAULT_MODEL, base_url=ollama_url, num_ctx=ollama_num_ctx
        )
    raise ValueError(f"Unknown provider {provider!r}; use 'anthropic' or 'ollama'.")
