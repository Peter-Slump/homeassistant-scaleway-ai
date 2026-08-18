"""Shared fixtures for the Scaleway AI test suite."""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import patch

from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
import pytest

pytest_plugins = ["pytest_homeassistant_custom_component"]


@pytest.fixture(autouse=True)
def _auto_enable_custom_integrations(
    enable_custom_integrations: None,
) -> None:
    """Enable loading of custom_components during tests."""


@pytest.fixture(autouse=True)
async def _bootstrap_ha_core(hass: HomeAssistant) -> None:
    """Bootstrap the `homeassistant` core component in every test.

    `conversation` (our declared dependency) registers a listener that
    looks up `homeassistant.exposed_entities` on start, which is populated
    by the `homeassistant` core component's setup. Without this every test
    fails with a `KeyError` on entry setup.
    """
    assert await async_setup_component(hass, "homeassistant", {})
    await hass.async_block_till_done()


@pytest.fixture
def mock_setup_entry() -> Generator[None]:
    """Patch async_setup_entry so config flow tests don't spin up the integration."""
    with patch(
        "custom_components.scaleway_ai.async_setup_entry", return_value=True
    ) as mocked:
        yield mocked
