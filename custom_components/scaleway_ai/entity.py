"""Shared LLM entity base for the Scaleway AI integration.

Handles the OpenAI-compatible `chat/completions` request/response cycle
against Scaleway's Generative APIs, including streaming and multi-round
tool calling into Home Assistant's `assist` LLM API.

Kept as its own module (rather than living inside `conversation.py`) so
future STT/TTS platforms can share the same client + tool-loop plumbing.

Design derived from `homeassistant.components.openai_conversation.entity`
(Apache-2.0), but the streaming transformer is rewritten for chat
completions — Scaleway does not currently expose the newer OpenAI
`/responses` API.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterable, Callable, Iterable
import json
import re
from typing import TYPE_CHECKING, Any

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.json import json_dumps
import openai
from openai import AsyncStream
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)

try:
    from probatio import to_openapi  # type: ignore[import-not-found]
except ImportError:  # Home Assistant < 2026.9
    from voluptuous_openapi import convert as to_openapi

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DOMAIN,
    ERROR_AUTH,
    ERROR_CANNOT_CONNECT,
    ERROR_CONTENT_FILTER,
    ERROR_RATE_LIMIT,
    ERROR_TRUNCATED,
    ERROR_UNKNOWN,
    LOGGER,
    MAX_TOOL_ITERATIONS,
)


class ScalewayChatError(HomeAssistantError):
    """Scaleway request failed; speak a localized plain-text message."""

    def __init__(self, error_key: str) -> None:
        """Store a stable error key instead of the raw API payload."""
        super().__init__(error_key)
        self.error_key = error_key


if TYPE_CHECKING:
    from . import ScalewayAIConfigEntry


# Scaleway rejects consecutive underscores/dashes (HA names like
# `homeassistant__GetLiveContext`) even though a single `_` or `-` is allowed.
_INVALID_FUNCTION_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_REPEATED_SEPARATORS = re.compile(r"[_-]{2,}")
_MAX_FUNCTION_NAME_LENGTH = 64


def _sanitize_function_name(name: str) -> str:
    """Rewrite a tool name so Scaleway's function-name validator accepts it."""
    cleaned = _INVALID_FUNCTION_CHARS.sub("_", name)
    cleaned = _REPEATED_SEPARATORS.sub("_", cleaned)
    cleaned = cleaned.strip("_-")
    if not cleaned:
        cleaned = "tool"
    if cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned[:_MAX_FUNCTION_NAME_LENGTH]


def _unique_sanitized_name(name: str, used: set[str]) -> str:
    """Sanitize `name` and append a suffix if that API name is already taken."""
    base = _sanitize_function_name(name)
    candidate = base
    index = 2
    while candidate in used:
        suffix = f"_{index}"
        candidate = f"{base[: _MAX_FUNCTION_NAME_LENGTH - len(suffix)]}{suffix}"
        index += 1
    used.add(candidate)
    return candidate


class _ToolNameMap:
    """Bidirectional map between Home Assistant tool names and Scaleway names."""

    def __init__(self, tool_names: Iterable[str] = ()) -> None:
        """Build the map from the tools exposed this turn."""
        self._to_api: dict[str, str] = {}
        self._from_api: dict[str, str] = {}
        used: set[str] = set()
        for name in tool_names:
            api_name = _unique_sanitized_name(name, used)
            self._to_api[name] = api_name
            self._from_api[api_name] = name

    def to_api(self, name: str) -> str:
        """Return the Scaleway-safe name for a Home Assistant tool."""
        if name in self._to_api:
            return self._to_api[name]
        api_name = _unique_sanitized_name(name, set(self._from_api))
        self._to_api[name] = api_name
        self._from_api[api_name] = name
        return api_name

    def from_api(self, name: str) -> str:
        """Return the Home Assistant tool name for a Scaleway function name."""
        return self._from_api.get(name, name)


def _format_tool(
    tool: llm.Tool,
    custom_serializer: Callable[[Any], Any] | None,
    names: _ToolNameMap,
) -> ChatCompletionToolParam:
    """Translate a HA `llm.Tool` into an OpenAI chat.completions tool schema."""
    parameters = to_openapi(tool.parameters, custom_serializer=custom_serializer)
    return {
        "type": "function",
        "function": {
            "name": names.to_api(tool.name),
            "description": tool.description or "",
            "parameters": parameters,
        },
    }


def _convert_content_to_messages(
    chat_content: Iterable[conversation.Content],
    names: _ToolNameMap | None = None,
) -> list[ChatCompletionMessageParam]:
    """Translate HA `ChatLog.content` into OpenAI chat.completions messages.

    ChatLog uses HA's role vocabulary (system / user / assistant / tool_result).
    OpenAI's chat.completions uses (system / user / assistant / tool). We map
    tool_result -> tool and encode assistant tool calls in the assistant
    message's `tool_calls` array.
    """
    messages: list[ChatCompletionMessageParam] = []

    for content in chat_content:
        if isinstance(content, conversation.SystemContent):
            if content.content:
                messages.append(
                    ChatCompletionSystemMessageParam(
                        role="system", content=content.content
                    )
                )
        elif isinstance(content, conversation.UserContent):
            messages.append(
                ChatCompletionUserMessageParam(role="user", content=content.content)
            )
        elif isinstance(content, conversation.AssistantContent):
            assistant_msg: ChatCompletionAssistantMessageParam = {"role": "assistant"}
            if content.content:
                assistant_msg["content"] = content.content
            if content.tool_calls:
                assistant_msg["tool_calls"] = [
                    ChatCompletionMessageToolCallParam(
                        id=tool_call.id,
                        type="function",
                        function={
                            "name": (
                                names.to_api(tool_call.tool_name)
                                if names
                                else tool_call.tool_name
                            ),
                            "arguments": json_dumps(tool_call.tool_args),
                        },
                    )
                    for tool_call in content.tool_calls
                ]
            messages.append(assistant_msg)
        elif isinstance(content, conversation.ToolResultContent):
            messages.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=content.tool_call_id,
                    content=json_dumps(content.tool_result),
                )
            )

    return messages


async def _transform_stream(  # noqa: PLR0912
    chat_log: conversation.ChatLog,
    stream: AsyncStream[ChatCompletionChunk],
    names: _ToolNameMap | None = None,
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Convert an OpenAI chat.completions stream into HA delta dicts.

    Tool-call fragments arrive in pieces keyed by `index`; we accumulate them
    and flush a single `tool_calls` delta once the model signals it's done
    (`finish_reason == "tool_calls"`).
    """
    started = False
    tool_call_buffers: dict[int, dict[str, str]] = {}

    async for chunk in stream:
        LOGGER.debug("Scaleway chat chunk: %s", chunk)

        if chunk.usage is not None:
            chat_log.async_trace(
                {
                    "stats": {
                        "input_tokens": chunk.usage.prompt_tokens,
                        "output_tokens": chunk.usage.completion_tokens,
                    }
                }
            )

        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        delta = choice.delta

        if not started:
            yield {"role": "assistant"}
            started = True

        if delta.content:
            yield {"content": delta.content}

        if delta.tool_calls:
            for tc in delta.tool_calls:
                buffer = tool_call_buffers.setdefault(
                    tc.index, {"id": "", "name": "", "arguments": ""}
                )
                if tc.id:
                    buffer["id"] = tc.id
                if tc.function is not None:
                    if tc.function.name:
                        buffer["name"] = tc.function.name
                    if tc.function.arguments:
                        buffer["arguments"] += tc.function.arguments

        finish_reason = choice.finish_reason
        if finish_reason == "tool_calls" and tool_call_buffers:
            tool_calls: list[llm.ToolInput] = []
            for _index, buf in sorted(tool_call_buffers.items()):
                try:
                    args = json.loads(buf["arguments"]) if buf["arguments"] else {}
                except json.JSONDecodeError as err:
                    LOGGER.warning(
                        "Scaleway returned malformed tool arguments for %s: %s",
                        buf["name"],
                        err,
                    )
                    args = {}
                tool_calls.append(
                    llm.ToolInput(
                        id=buf["id"],
                        tool_name=names.from_api(buf["name"]) if names else buf["name"],
                        tool_args=args,
                    )
                )
            yield {"tool_calls": tool_calls}
            tool_call_buffers.clear()
        elif finish_reason == "length":
            raise ScalewayChatError(ERROR_TRUNCATED)
        elif finish_reason == "content_filter":
            raise ScalewayChatError(ERROR_CONTENT_FILTER)


class ScalewayAIBaseLLMEntity(Entity):
    """Shared base for Scaleway AI LLM-backed entities."""

    _attr_has_entity_name = True
    _attr_name: str | None = None

    def __init__(
        self,
        entry: ScalewayAIConfigEntry,
        subentry: ConfigSubentry,
    ) -> None:
        """Initialise the entity from a parent entry + subentry pair."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="Scaleway",
            model=subentry.data.get(CONF_CHAT_MODEL, DEFAULT_MODEL),
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        max_iterations: int = MAX_TOOL_ITERATIONS,
    ) -> None:
        """Drive one user turn to completion, looping through tool calls."""
        options = self.subentry.data
        client = self.entry.runtime_data

        model = options.get(CONF_CHAT_MODEL, DEFAULT_MODEL)
        temperature = options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        top_p = options.get(CONF_TOP_P, DEFAULT_TOP_P)
        max_tokens = options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)

        names = (
            _ToolNameMap(t.name for t in chat_log.llm_api.tools)
            if chat_log.llm_api
            else _ToolNameMap()
        )

        tools: list[ChatCompletionToolParam] | None = None
        if chat_log.llm_api:
            tools = [
                _format_tool(t, chat_log.llm_api.custom_serializer, names)
                for t in chat_log.llm_api.tools
            ]

        for _iteration in range(max_iterations):
            messages = _convert_content_to_messages(chat_log.content, names)

            request_args: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
                "user": chat_log.conversation_id,
            }
            if tools:
                request_args["tools"] = tools

            try:
                stream = await client.chat.completions.create(**request_args)
                content_stream: AsyncIterable[
                    conversation.AssistantContent | conversation.ToolResultContent
                ] = chat_log.async_add_delta_content_stream(
                    self.entity_id,
                    _transform_stream(chat_log, stream, names),
                )
                _ = [content async for content in content_stream]
            except ScalewayChatError:
                raise
            except openai.AuthenticationError as err:
                LOGGER.error("Scaleway rejected our API key: %s", err)
                raise ScalewayChatError(ERROR_AUTH) from err
            except openai.RateLimitError as err:
                LOGGER.error("Scaleway rate-limited request: %s", err)
                raise ScalewayChatError(ERROR_RATE_LIMIT) from err
            except openai.APIConnectionError as err:
                LOGGER.error("Could not reach Scaleway: %s", err)
                raise ScalewayChatError(ERROR_CANNOT_CONNECT) from err
            except openai.OpenAIError as err:
                LOGGER.error("Error talking to Scaleway: %s", err)
                raise ScalewayChatError(ERROR_UNKNOWN) from err

            if not chat_log.unresponded_tool_results:
                break


__all__ = [
    "ScalewayAIBaseLLMEntity",
    "ScalewayChatError",
    "_ToolNameMap",
    "_convert_content_to_messages",
    "_format_tool",
    "_sanitize_function_name",
    "_transform_stream",
]
