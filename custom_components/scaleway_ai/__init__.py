"""The Scaleway AI integration.

Bridges Home Assistant's Assist pipeline to Scaleway's Generative APIs
(https://www.scaleway.com/en/generative-apis/), an OpenAI-compatible LLM
inference service hosted in `fr-par`.

Adapted in shape from `homeassistant.components.openai_conversation`
(Apache-2.0). Adapted heavily: Scaleway uses the classic `chat/completions`
API rather than OpenAI's newer `/responses` API, so the streaming transformer
and tool loop are rewritten from scratch — but the overall entry/subentry
layout mirrors the upstream integration so users familiar with it feel at
home.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_PROMPT, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv, llm
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType
import openai

from .const import (
    CONF_BASE_URL,
    CONF_PROJECT_ID,
    DEFAULT_BASE_URL,
    DOMAIN,
    LOGGER,
    SUBENTRY_TYPE_CONVERSATION,
)

if TYPE_CHECKING:
    pass

PLATFORMS: tuple[Platform, ...] = (Platform.CONVERSATION,)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

type ScalewayAIConfigEntry = ConfigEntry[openai.AsyncOpenAI]


def _build_client(hass: HomeAssistant, entry: ConfigEntry) -> openai.AsyncOpenAI:
    """Instantiate an OpenAI-compatible async client pointed at Scaleway."""
    base_url = entry.data.get(CONF_BASE_URL, DEFAULT_BASE_URL)
    project_id = entry.data.get(CONF_PROJECT_ID)
    if project_id and base_url == DEFAULT_BASE_URL:
        base_url = f"https://api.scaleway.ai/{project_id}/v1"

    return openai.AsyncOpenAI(
        api_key=entry.data[CONF_API_KEY],
        base_url=base_url,
        http_client=get_async_client(hass),
    )


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Scaleway AI integration (no YAML config)."""
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ScalewayAIConfigEntry) -> bool:
    """Set up Scaleway AI from a config entry."""
    client = _build_client(hass, entry)

    try:
        await client.with_options(timeout=10.0).models.list()
    except openai.AuthenticationError as err:
        raise ConfigEntryAuthFailed(err) from err
    except openai.OpenAIError as err:
        raise ConfigEntryNotReady(err) from err

    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_update_options))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ScalewayAIConfigEntry) -> bool:
    """Unload a Scaleway AI config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def _async_update_options(
    hass: HomeAssistant, entry: ScalewayAIConfigEntry
) -> None:
    """Reload the entry when options change."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_migrate_entry(
    hass: HomeAssistant, entry: ScalewayAIConfigEntry
) -> bool:
    """Migrate old entries. v0.1 ships at VERSION=1 so this is a no-op stub."""
    LOGGER.debug(
        "Migrating from version %s.%s (no migrations defined yet)",
        entry.version,
        entry.minor_version,
    )
    return True


def _default_conversation_subentry() -> ConfigSubentry:
    """Build the default conversation subentry created on first setup."""
    from .const import DEFAULT_MODEL, DEFAULT_TEMPERATURE

    return ConfigSubentry(
        data=MappingProxyType(
            {
                "chat_model": DEFAULT_MODEL,
                "temperature": DEFAULT_TEMPERATURE,
                CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
                CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
            }
        ),
        subentry_type=SUBENTRY_TYPE_CONVERSATION,
        title="Scaleway AI Conversation",
        unique_id=None,
    )
