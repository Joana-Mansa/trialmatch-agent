"""A scripted stand-in for ``anthropic.Anthropic`` that returns real SDK types.

``FakeAnthropic(script).beta.messages.create(**kwargs)`` records a deep copy of the
kwargs and returns the next scripted response, built with ``BetaMessage.model_validate``
so the agent code runs against the same classes the real SDK returns. A script item
can be a dict (validated into a BetaMessage), a ready BetaMessage, an exception
(raised), or a callable taking the kwargs and returning one of those.
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Callable
from typing import Any

from anthropic.types.beta import BetaMessage

_ids = itertools.count(1)


def text(t: str) -> dict[str, Any]:
    return {"type": "text", "text": t}


def thinking(t: str = "") -> dict[str, Any]:
    return {"type": "thinking", "thinking": t, "signature": f"sig-{next(_ids)}"}


def tool_use(name: str, tool_input: dict[str, Any], id: str | None = None) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": id or f"toolu_{next(_ids):04d}",
        "name": name,
        "input": tool_input,
    }


def fallback_block(frm: str = "claude-opus-5-5", to: str = "claude-opus-5") -> dict[str, Any]:
    return {
        "type": "fallback",
        "from": {"model": frm},
        "to": {"model": to},
        "trigger": {"type": "refusal", "category": "cyber"},
    }


def message(
    content: list[dict[str, Any]],
    stop_reason: str,
    *,
    stop_details: dict[str, Any] | None = None,
    iterations: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
    model: str = "claude-opus-5-5",
) -> dict[str, Any]:
    u = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 80,
        "cache_creation_input_tokens": 20,
    }
    u.update(usage or {})
    if iterations is not None:
        u["iterations"] = iterations
    return {
        "id": f"msg_{next(_ids):04d}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "stop_details": stop_details,
        "usage": u,
    }


def tool_turn(*calls: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    """An assistant turn with a thinking block followed by one tool_use per call."""
    return message([thinking(), *[tool_use(n, i) for n, i in calls]], "tool_use")


ScriptItem = dict[str, Any] | BetaMessage | BaseException | Callable[[dict[str, Any]], Any]


class _FakeMessages:
    def __init__(self, script: list[ScriptItem]):
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> BetaMessage:
        self.calls.append(copy.deepcopy(kwargs))
        if not self._script:
            raise AssertionError("FakeAnthropic script exhausted: unexpected extra API call")
        item = self._script.pop(0)
        if callable(item) and not isinstance(item, BaseException | BetaMessage):
            item = item(kwargs)
        if isinstance(item, BaseException):
            raise item
        msg = item if isinstance(item, BetaMessage) else BetaMessage.model_validate(item)
        # The SDK sets this from the request-id header on real responses.
        msg._request_id = f"req_{len(self.calls):03d}"
        return msg

    @property
    def remaining(self) -> int:
        return len(self._script)


class _FakeBeta:
    def __init__(self, messages: _FakeMessages):
        self.messages = messages


class FakeAnthropic:
    def __init__(self, script: list[ScriptItem]):
        self.beta = _FakeBeta(_FakeMessages(script))

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.beta.messages.calls
