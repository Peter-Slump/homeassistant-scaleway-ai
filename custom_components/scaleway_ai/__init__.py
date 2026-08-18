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
from typing import TYPE_CHECKING, Any, cast

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_API_KEY, CONF_PROMPT, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType
import openai

from .const import (
    CONF_BASE_URL,
    CONF_PROJECT_ID,
    CONF_STT_MODEL,
    DEFAULT_BASE_URL,
    DEFAULT_STT_MODEL,
    DEFAULT_STT_NAME,
    DEFAULT_STT_PROMPT,
    DOMAIN,
    LOGGER,
    SUBENTRY_TYPE_STT,
)

if TYPE_CHECKING:
    pass

PLATFORMS: tuple[Platform, ...] = (Platform.CONVERSATION, Platform.STT)

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
        http_client=cast(Any, get_async_client(hass)),
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
    """Migrate config entries when integration structure changes."""
    LOGGER.debug(
        "Migrating from version %s.%s",
        entry.version,
        entry.minor_version,
    )

    if entry.version == 1 and entry.minor_version == 1:
        has_stt = any(
            subentry.subentry_type == SUBENTRY_TYPE_STT
            for subentry in entry.subentries.values()
        )
        if not has_stt:
            hass.config_entries.async_add_subentry(
                entry,
                ConfigSubentry(
                    data=MappingProxyType(_default_stt_subentry_data()),
                    subentry_type=SUBENTRY_TYPE_STT,
                    title=DEFAULT_STT_NAME,
                    unique_id=None,
                ),
            )
        hass.config_entries.async_update_entry(entry, minor_version=2)

    return True


def _default_stt_subentry_data() -> dict[str, str]:
    """Default data for a new speech-to-text subentry."""
    return {
        CONF_STT_MODEL: DEFAULT_STT_MODEL,
        CONF_PROMPT: DEFAULT_STT_PROMPT,
    }
