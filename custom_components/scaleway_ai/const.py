"""Constants for the Scaleway AI integration."""

from __future__ import annotations

import logging
from typing import Final

DOMAIN: Final = "scaleway_ai"

LOGGER: Final = logging.getLogger(__package__)

CONF_API_KEY: Final = "api_key"
CONF_PROJECT_ID: Final = "project_id"
CONF_BASE_URL: Final = "base_url"

CONF_CHAT_MODEL: Final = "chat_model"
CONF_TEMPERATURE: Final = "temperature"
CONF_MAX_TOKENS: Final = "max_tokens"
CONF_TOP_P: Final = "top_p"

DEFAULT_BASE_URL: Final = "https://api.scaleway.ai/v1"

# Default model: EU-native, cheap, strong multilingual (incl. Dutch),
# solid tool calling. Users can pick any listed by GET /v1/models at setup.
DEFAULT_MODEL: Final = "mistral-small-3.2-24b-instruct-2506"

DEFAULT_TEMPERATURE: Final = 0.7
DEFAULT_MAX_TOKENS: Final = 1024
DEFAULT_TOP_P: Final = 1.0

# Cap on tool call round-trips per user turn, mirrors HA core's OpenAI
# integration. Protects against infinite tool loops.
MAX_TOOL_ITERATIONS: Final = 8

# Fallback list of chat models exposed if GET /v1/models fails during setup.
# Verified from the Scaleway Generative APIs catalog as of Aug 2026.
FALLBACK_CHAT_MODELS: Final[tuple[str, ...]] = (
    "mistral-small-3.2-24b-instruct-2506",
    "mistral-medium-3.5-128b",
    "llama-3.3-70b-instruct",
    "gpt-oss-120b",
    "qwen3-235b-a22b-instruct-2507",
    "deepseek-v4-flash-0731",
    "gemma-4-26b-a4b-it",
    "glm-5.2",
)

SUBENTRY_TYPE_CONVERSATION: Final = "conversation"
