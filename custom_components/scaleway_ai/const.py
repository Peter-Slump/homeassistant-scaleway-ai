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
CONF_STT_MODEL: Final = "stt_model"
CONF_TEMPERATURE: Final = "temperature"
CONF_MAX_TOKENS: Final = "max_tokens"
CONF_TOP_P: Final = "top_p"
CONF_LANGUAGE: Final = "language"
CONF_LOG_CONVERSATION: Final = "log_conversation"
DEFAULT_LOG_CONVERSATION: Final = False

LANG_EN: Final = "en"
LANG_NL: Final = "nl"
DEFAULT_LANGUAGE: Final = LANG_EN

ERROR_AUTH: Final = "auth"
ERROR_RATE_LIMIT: Final = "rate_limit"
ERROR_CANNOT_CONNECT: Final = "cannot_connect"
ERROR_TRUNCATED: Final = "truncated"
ERROR_CONTENT_FILTER: Final = "content_filter"
ERROR_UNKNOWN: Final = "unknown"

# Short, speakable replies. Never include the raw Scaleway/OpenAI JSON body.
SPOKEN_ERRORS: Final[dict[str, dict[str, str]]] = {
    LANG_EN: {
        ERROR_AUTH: (
            "Sorry, Scaleway rejected the API key. Please reconfigure the integration."
        ),
        ERROR_RATE_LIMIT: (
            "Sorry, Scaleway is rate limiting requests. Please try again shortly."
        ),
        ERROR_CANNOT_CONNECT: "Sorry, I could not reach Scaleway. Please try again.",
        ERROR_TRUNCATED: "Sorry, the answer was cut off because it was too long.",
        ERROR_CONTENT_FILTER: "Sorry, Scaleway blocked that response.",
        ERROR_UNKNOWN: (
            "Sorry, something went wrong talking to Scaleway. Please try again."
        ),
    },
    LANG_NL: {
        ERROR_AUTH: (
            "Sorry, Scaleway heeft de API-sleutel geweigerd. "
            "Configureer de integratie opnieuw."
        ),
        ERROR_RATE_LIMIT: (
            "Sorry, Scaleway beperkt het aantal verzoeken. "
            "Probeer het zo meteen opnieuw."
        ),
        ERROR_CANNOT_CONNECT: (
            "Sorry, ik kon Scaleway niet bereiken. Probeer het opnieuw."
        ),
        ERROR_TRUNCATED: "Sorry, het antwoord is afgekapt omdat het te lang was.",
        ERROR_CONTENT_FILTER: "Sorry, Scaleway heeft dat antwoord geblokkeerd.",
        ERROR_UNKNOWN: (
            "Sorry, er ging iets mis bij het praten met Scaleway. Probeer het opnieuw."
        ),
    },
}


def spoken_error(language: str | None, key: str) -> str:
    """Return a speakable error for the configured language."""
    messages = SPOKEN_ERRORS.get(language or "", SPOKEN_ERRORS[DEFAULT_LANGUAGE])
    return messages.get(key, messages[ERROR_UNKNOWN])

DEFAULT_BASE_URL: Final = "https://api.scaleway.ai/v1"

# Default model: EU-native, cheap, strong multilingual (incl. Dutch),
# solid tool calling. Users can pick any listed by GET /v1/models at setup.
DEFAULT_MODEL: Final = "mistral-small-3.2-24b-instruct-2506"

# Scaleway serverless STT model (OpenAI-compatible /v1/audio/transcriptions).
DEFAULT_STT_MODEL: Final = "whisper-large-v3"

DEFAULT_STT_PROMPT: Final = (
    "The following conversation is a smart home user talking to Home Assistant."
)

DEFAULT_STT_NAME: Final = "Scaleway AI STT"

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

FALLBACK_STT_MODELS: Final[tuple[str, ...]] = (
    "whisper-large-v3",
    "voxtral-small-24b-2507",
)

SUBENTRY_TYPE_CONVERSATION: Final = "conversation"
SUBENTRY_TYPE_STT: Final = "stt"
