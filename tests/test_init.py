"""Basic setup/unload smoke test for the Scaleway AI integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.scaleway_ai.const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    DEFAULT_BASE_URL,
    DOMAIN,
    SUBENTRY_TYPE_STT,
)


async def test_setup_and_unload(hass: HomeAssistant) -> None:
    """Config entry sets up and unloads cleanly when Scaleway responds OK."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_KEY: "scw-test-key", CONF_BASE_URL: DEFAULT_BASE_URL},
        title="Scaleway AI",
        version=1,
        minor_version=1,
    )
    entry.add_to_hass(hass)

    with patch("custom_components.scaleway_ai.openai.AsyncOpenAI") as mock_client_cls:
        client = mock_client_cls.return_value
        client.with_options.return_value.models.list = AsyncMock(return_value=None)
        client.chat.completions.create = AsyncMock()

        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

        assert entry.state is ConfigEntryState.LOADED

        assert await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()

        assert entry.state is ConfigEntryState.NOT_LOADED


async def test_migration_adds_stt_subentry(hass: HomeAssistant) -> None:
    """v1.1 entries get an STT subentry when upgrading to v0.2."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_KEY: "scw-test-key", CONF_BASE_URL: DEFAULT_BASE_URL},
        title="Scaleway AI",
        version=1,
        minor_version=1,
    )
    entry.add_to_hass(hass)

    with patch("custom_components.scaleway_ai.openai.AsyncOpenAI") as mock_client_cls:
        client = mock_client_cls.return_value
        client.with_options.return_value.models.list = AsyncMock(return_value=None)
        client.chat.completions.create = AsyncMock()
        client.audio.transcriptions.create = AsyncMock()

        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    assert entry.minor_version == 2
    subentry_types = {s.subentry_type for s in entry.subentries.values()}
    assert SUBENTRY_TYPE_STT in subentry_types
