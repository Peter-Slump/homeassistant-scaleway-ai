"""Smoke tests for the streaming transformer and message conversion."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.scaleway_ai.const import (
    CONF_LANGUAGE,
    ERROR_CONTENT_FILTER,
    ERROR_TRUNCATED,
    ERROR_UNKNOWN,
    LANG_EN,
    LANG_NL,
    spoken_error,
)
from custom_components.scaleway_ai.conversation import ScalewayAIConversationEntity
from custom_components.scaleway_ai.entity import (
    ScalewayChatError,
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


def test_spoken_error_is_plain_text_in_english_and_dutch() -> None:
    """Voice replies stay speakable and never include the raw JSON payload."""
    english = spoken_error(LANG_EN, ERROR_UNKNOWN)
    dutch = spoken_error(LANG_NL, ERROR_UNKNOWN)
    assert english != dutch
    assert "{" not in english
    assert "{" not in dutch
    assert spoken_error("de", ERROR_UNKNOWN) == english
    assert spoken_error(None, "missing") == english


@pytest.mark.asyncio
async def test_transform_stream_truncated_raises_spoken_error() -> None:
    """A max-token cutoff becomes a stable error key, not a JSON string."""
    chat_log = MagicMock()
    stream = _as_stream([_fake_chunk(finish_reason="length")])

    with pytest.raises(ScalewayChatError) as err:
        _ = [d async for d in _transform_stream(chat_log, stream)]
    assert err.value.error_key == ERROR_TRUNCATED
    assert "{" not in str(err.value)


@pytest.mark.asyncio
async def test_transform_stream_content_filter_raises_spoken_error() -> None:
    """A content-filter stop becomes a stable error key."""
    chat_log = MagicMock()
    stream = _as_stream([_fake_chunk(finish_reason="content_filter")])

    with pytest.raises(ScalewayChatError) as err:
        _ = [d async for d in _transform_stream(chat_log, stream)]
    assert err.value.error_key == ERROR_CONTENT_FILTER


@pytest.mark.parametrize(
    ("language", "error_key"),
    [(LANG_EN, ERROR_UNKNOWN), (LANG_NL, ERROR_UNKNOWN)],
)
def test_spoken_error_result_uses_configured_language(
    language: str, error_key: str
) -> None:
    """Conversation errors become IntentResponse speech in the chosen language."""
    entity = ScalewayAIConversationEntity.__new__(ScalewayAIConversationEntity)
    entity.subentry = MagicMock()
    entity.subentry.data = {CONF_LANGUAGE: language}
    user_input = MagicMock()
    user_input.conversation_id = "conv-1"

    result = entity._spoken_error_result(user_input, error_key)

    assert result.conversation_id == "conv-1"
    speech = result.response.speech["plain"]["speech"]
    assert speech == spoken_error(language, error_key)
    assert "{" not in speech


@pytest.mark.asyncio
async def test_handle_message_speaks_scaleway_error_instead_of_json() -> None:
    """API failures become a Dutch/English sentence, not the raw error body."""
    entity = ScalewayAIConversationEntity.__new__(ScalewayAIConversationEntity)
    entity.subentry = MagicMock()
    entity.subentry.data = {CONF_LANGUAGE: LANG_NL}
    user_input = MagicMock()
    user_input.conversation_id = "conv-1"
    user_input.extra_system_prompt = None
    user_input.as_llm_context = MagicMock(return_value="ctx")
    chat_log = MagicMock()
    chat_log.async_provide_llm_data = AsyncMock()
    entity._async_handle_chat_log = AsyncMock(  # type: ignore[method-assign]
        side_effect=ScalewayChatError(ERROR_UNKNOWN)
    )

    result = await entity._async_handle_message(user_input, chat_log)

    assert result.response.speech["plain"]["speech"] == spoken_error(
        LANG_NL, ERROR_UNKNOWN
    )
    assert "{" not in result.response.speech["plain"]["speech"]
