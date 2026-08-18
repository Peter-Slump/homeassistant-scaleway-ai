"""Smoke tests for the streaming transformer and message conversion."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from custom_components.scaleway_ai.entity import (
    _convert_content_to_messages,
    _transform_stream,
)


def _fake_chunk(
    *,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> Any:
    """Build a duck-typed ChatCompletionChunk stand-in."""
    chunk = MagicMock()
    if usage is not None:
        usage_obj = MagicMock()
        usage_obj.prompt_tokens = usage["prompt_tokens"]
        usage_obj.completion_tokens = usage["completion_tokens"]
        chunk.usage = usage_obj
    else:
        chunk.usage = None

    if content is None and tool_calls is None and finish_reason is None:
        chunk.choices = []
        return chunk

    choice = MagicMock()
    choice.finish_reason = finish_reason

    delta = MagicMock()
    delta.content = content
    if tool_calls is not None:
        deltas = []
        for tc in tool_calls:
            tc_obj = MagicMock()
            tc_obj.index = tc["index"]
            tc_obj.id = tc.get("id", "")
            function = MagicMock()
            function.name = tc.get("name")
            function.arguments = tc.get("arguments")
            tc_obj.function = function
            deltas.append(tc_obj)
        delta.tool_calls = deltas
    else:
        delta.tool_calls = None
    choice.delta = delta
    chunk.choices = [choice]
    return chunk


async def _as_stream(chunks: list[Any]) -> AsyncIterator[Any]:
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_transform_stream_content_only() -> None:
    """Plain content chunks yield an assistant role delta then content deltas."""
    chat_log = MagicMock()
    stream = _as_stream(
        [
            _fake_chunk(content="Hel"),
            _fake_chunk(content="lo!"),
            _fake_chunk(
                finish_reason="stop",
                usage={"prompt_tokens": 5, "completion_tokens": 2},
            ),
        ]
    )

    deltas = [d async for d in _transform_stream(chat_log, stream)]
    assert deltas[0] == {"role": "assistant"}
    assert {"content": "Hel"} in deltas
    assert {"content": "lo!"} in deltas
    chat_log.async_trace.assert_called_once()


@pytest.mark.asyncio
async def test_transform_stream_tool_call_assembly() -> None:
    """Fragmented tool call arguments assemble into a single tool_calls delta."""
    chat_log = MagicMock()
    first_tc = {
        "index": 0,
        "id": "call_1",
        "name": "HassTurnOn",
        "arguments": '{"na',
    }
    stream = _as_stream(
        [
            _fake_chunk(tool_calls=[first_tc]),
            _fake_chunk(tool_calls=[{"index": 0, "arguments": 'me":"kitchen"}'}]),
            _fake_chunk(finish_reason="tool_calls"),
        ]
    )

    deltas = [d async for d in _transform_stream(chat_log, stream)]
    tool_deltas = [d for d in deltas if "tool_calls" in d]
    assert len(tool_deltas) == 1
    calls = tool_deltas[0]["tool_calls"]
    assert len(calls) == 1
    assert calls[0].tool_name == "HassTurnOn"
    assert calls[0].tool_args == {"name": "kitchen"}
    assert calls[0].id == "call_1"


def test_convert_content_roundtrip() -> None:
    """SystemContent/UserContent/AssistantContent map to expected OpenAI shapes."""
    from homeassistant.components import conversation
    from homeassistant.helpers import llm

    contents = [
        conversation.SystemContent(content="You are helpful."),
        conversation.UserContent(content="Turn on the lights."),
        conversation.AssistantContent(
            agent_id="scaleway_ai.conv",
            content=None,
            tool_calls=[
                llm.ToolInput(
                    id="call_1",
                    tool_name="HassTurnOn",
                    tool_args={"name": "kitchen"},
                )
            ],
        ),
        conversation.ToolResultContent(
            agent_id="scaleway_ai.conv",
            tool_call_id="call_1",
            tool_name="HassTurnOn",
            tool_result={"success": True},
        ),
    ]
    messages = _convert_content_to_messages(contents)

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["tool_calls"][0]["function"]["name"] == "HassTurnOn"
    assert messages[3]["role"] == "tool"
    assert messages[3]["tool_call_id"] == "call_1"
