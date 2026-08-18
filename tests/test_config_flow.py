"""Config flow tests for the Scaleway AI integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
import httpx
import openai
import pytest

from custom_components.scaleway_ai.const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    DEFAULT_BASE_URL,
    DOMAIN,
    SUBENTRY_TYPE_CONVERSATION,
)


def _make_openai_error(exc_cls: type[openai.OpenAIError]) -> openai.OpenAIError:
    """Build a properly-constructed OpenAI error instance for testing."""
    request = httpx.Request("GET", "https://api.scaleway.ai/v1/models")
    if exc_cls is openai.AuthenticationError:
        response = httpx.Response(401, request=request)
        return openai.AuthenticationError("bad key", response=response, body=None)
    if exc_cls is openai.APIConnectionError:
        return openai.APIConnectionError(request=request)
    raise NotImplementedError(f"No factory for {exc_cls}")


async def test_user_flow_happy_path(
    hass: HomeAssistant,
    mock_setup_entry: AsyncMock,
) -> None:
    """Valid API key creates an entry with a default conversation subentry."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "user"

    with patch(
        "custom_components.scaleway_ai.config_flow.openai.AsyncOpenAI"
    ) as mock_client_cls:
        client = mock_client_cls.return_value
        client.with_options.return_value.models.list = AsyncMock(return_value=None)

        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {CONF_API_KEY: "scw-valid", CONF_BASE_URL: DEFAULT_BASE_URL},
        )
        await hass.async_block_till_done()

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == "Scaleway AI"
    entry = result["result"]
    subentry_types = {s.subentry_type for s in entry.subentries.values()}
    assert SUBENTRY_TYPE_CONVERSATION in subentry_types


@pytest.mark.parametrize(
    ("exc", "expected_error"),
    [
        (openai.AuthenticationError, "invalid_auth"),
        (openai.APIConnectionError, "cannot_connect"),
    ],
)
async def test_user_flow_errors(
    hass: HomeAssistant,
    mock_setup_entry: AsyncMock,
    exc: type[openai.OpenAIError],
    expected_error: str,
) -> None:
    """Auth + connection errors surface as translatable form errors."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    with patch(
        "custom_components.scaleway_ai.config_flow.openai.AsyncOpenAI"
    ) as mock_client_cls:
        client = mock_client_cls.return_value
        client.with_options.return_value.models.list = AsyncMock(
            side_effect=_make_openai_error(exc)
        )

        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {CONF_API_KEY: "scw-bad", CONF_BASE_URL: DEFAULT_BASE_URL},
        )

    assert result["type"] == FlowResultType.FORM
    assert result["errors"] == {"base": expected_error}
