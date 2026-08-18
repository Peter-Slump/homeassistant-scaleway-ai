"""Config + subentry flows for the Scaleway AI integration.

Parent config entry captures Scaleway credentials (API key, optional project
ID, optional base URL override for Managed Inference endpoints). Subentries
of type `conversation` configure individual assistant personas.

Shape mirrors `homeassistant.components.openai_conversation.config_flow`
(Apache-2.0) with Scaleway-specific fields and simplified model options
(no reasoning-effort / verbosity / web-search / etc.).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from homeassistant.config_entries import (
    SOURCE_REAUTH,
    ConfigEntry,
    ConfigEntryState,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_NAME, CONF_PROMPT
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)
from homeassistant.helpers.typing import VolDictType
import openai
import voluptuous as vol

from .const import (
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROJECT_ID,
    CONF_STT_MODEL,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_BASE_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_STT_MODEL,
    DEFAULT_STT_NAME,
    DEFAULT_STT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DOMAIN,
    FALLBACK_CHAT_MODELS,
    FALLBACK_STT_MODELS,
    LOGGER,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_STT,
)

CONF_ADVANCED = "advanced"

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): str,
        vol.Optional(CONF_PROJECT_ID): str,
        vol.Optional(CONF_BASE_URL, default=DEFAULT_BASE_URL): str,
    }
)

STEP_REAUTH_DATA_SCHEMA = vol.Schema({vol.Required(CONF_API_KEY): str})

RECOMMENDED_CONVERSATION_OPTIONS: dict[str, Any] = {
    CONF_CHAT_MODEL: DEFAULT_MODEL,
    CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
    CONF_TOP_P: DEFAULT_TOP_P,
    CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
}

DEFAULT_CONVERSATION_NAME = "Scaleway AI Conversation"

RECOMMENDED_STT_OPTIONS: dict[str, Any] = {
    CONF_STT_MODEL: DEFAULT_STT_MODEL,
    CONF_PROMPT: DEFAULT_STT_PROMPT,
}


async def _validate_credentials(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Attempt a `GET /v1/models` to verify the API key and endpoint."""
    base_url = data.get(CONF_BASE_URL, DEFAULT_BASE_URL)
    project_id = data.get(CONF_PROJECT_ID)
    if project_id and base_url == DEFAULT_BASE_URL:
        base_url = f"https://api.scaleway.ai/{project_id}/v1"

    client = openai.AsyncOpenAI(
        api_key=data[CONF_API_KEY],
        base_url=base_url,
        http_client=cast(Any, get_async_client(hass)),
    )
    await client.with_options(timeout=10.0).models.list()


async def _list_chat_models(
    hass: HomeAssistant, entry_data: Mapping[str, Any]
) -> list[str]:
    """Fetch the live model catalog; fall back to a hardcoded shortlist."""
    base_url = entry_data.get(CONF_BASE_URL, DEFAULT_BASE_URL)
    project_id = entry_data.get(CONF_PROJECT_ID)
    if project_id and base_url == DEFAULT_BASE_URL:
        base_url = f"https://api.scaleway.ai/{project_id}/v1"

    try:
        client = openai.AsyncOpenAI(
            api_key=entry_data[CONF_API_KEY],
            base_url=base_url,
            http_client=cast(Any, get_async_client(hass)),
        )
        result = await client.with_options(timeout=10.0).models.list()
        return sorted(m.id for m in result.data)
    except openai.OpenAIError as err:
        LOGGER.warning(
            "Falling back to hardcoded model list (Scaleway /models failed: %s)", err
        )
        return list(FALLBACK_CHAT_MODELS)


async def _list_stt_models(
    hass: HomeAssistant, entry_data: Mapping[str, Any]
) -> list[str]:
    """Fetch STT models from Scaleway; fall back to a hardcoded shortlist."""
    all_models = await _list_chat_models(hass, entry_data)
    stt_models = [
        model
        for model in all_models
        if "whisper" in model or "voxtral" in model or model in FALLBACK_STT_MODELS
    ]
    if not stt_models:
        return list(FALLBACK_STT_MODELS)
    return stt_models


class ScalewayAIConfigFlow(ConfigFlow, domain=DOMAIN):
    """Parent config flow for Scaleway AI credentials."""

    VERSION = 1
    MINOR_VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial credentials step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self._async_abort_entries_match({CONF_API_KEY: user_input[CONF_API_KEY]})
            try:
                await _validate_credentials(self.hass, user_input)
            except openai.AuthenticationError:
                errors["base"] = "invalid_auth"
            except openai.APIConnectionError:
                errors["base"] = "cannot_connect"
            except openai.OpenAIError:
                LOGGER.exception("Unexpected error validating Scaleway credentials")
                errors["base"] = "unknown"
            else:
                if self.source == SOURCE_REAUTH:
                    return self.async_update_reload_and_abort(
                        self._get_reauth_entry(), data_updates=user_input
                    )
                return self.async_create_entry(
                    title="Scaleway AI",
                    data=user_input,
                    subentries=[
                        {
                            "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                            "data": RECOMMENDED_CONVERSATION_OPTIONS,
                            "title": DEFAULT_CONVERSATION_NAME,
                            "unique_id": None,
                        },
                        {
                            "subentry_type": SUBENTRY_TYPE_STT,
                            "data": RECOMMENDED_STT_OPTIONS,
                            "title": DEFAULT_STT_NAME,
                            "unique_id": None,
                        },
                    ],
                )

        return self.async_show_form(
            step_id="user",
            data_schema=self.add_suggested_values_to_schema(
                STEP_USER_DATA_SCHEMA, user_input
            ),
            errors=errors,
        )

    async def async_step_reauth(
        self, entry_data: Mapping[str, Any]
    ) -> ConfigFlowResult:
        """Kick off reauthentication when credentials go stale."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Collect a fresh API key on reauth."""
        if user_input is None:
            return self.async_show_form(
                step_id="reauth_confirm", data_schema=STEP_REAUTH_DATA_SCHEMA
            )
        existing = self._get_reauth_entry().data
        merged = {**existing, **user_input}
        return await self.async_step_user(merged)

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Advertise the subentry types this integration supports."""
        return {
            SUBENTRY_TYPE_CONVERSATION: ConversationSubentryFlow,
            SUBENTRY_TYPE_STT: STTSubentryFlow,
        }


class ConversationSubentryFlow(ConfigSubentryFlow):
    """Add/reconfigure a conversation persona under a Scaleway AI entry."""

    def __init__(self) -> None:
        """Init mutable state carried between steps."""
        self._options: dict[str, Any] = {}

    @property
    def _is_new(self) -> bool:
        """True when this is a fresh subentry (vs. a reconfigure)."""
        return self.source == "user"

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Entry point for creating a new conversation subentry."""
        self._options = {**RECOMMENDED_CONVERSATION_OPTIONS}
        return await self.async_step_init()

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Entry point for editing an existing subentry."""
        self._options = dict(self._get_reconfigure_subentry().data)
        return await self.async_step_init()

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Name + prompt + LLM API step."""
        entry = self._get_entry()
        if entry.state is not ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        options = self._options
        hass_apis: list[SelectOptionDict] = [
            SelectOptionDict(label=api.name, value=api.id)
            for api in llm.async_get_apis(self.hass)
        ]

        if suggested := options.get(CONF_LLM_HASS_API):
            if isinstance(suggested, str):
                suggested = [suggested]
            valid = {api.id for api in llm.async_get_apis(self.hass)}
            options[CONF_LLM_HASS_API] = [a for a in suggested if a in valid]

        step_schema: VolDictType = {}
        if self._is_new:
            step_schema[vol.Required(CONF_NAME, default=DEFAULT_CONVERSATION_NAME)] = (
                str
            )

        step_schema.update(
            {
                vol.Optional(
                    CONF_PROMPT,
                    description={
                        "suggested_value": options.get(
                            CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                        )
                    },
                ): TemplateSelector(),
                vol.Optional(CONF_LLM_HASS_API): SelectSelector(
                    SelectSelectorConfig(options=hass_apis, multiple=True)
                ),
            }
        )

        if user_input is not None:
            if not user_input.get(CONF_LLM_HASS_API):
                user_input.pop(CONF_LLM_HASS_API, None)
                options.pop(CONF_LLM_HASS_API, None)
            options.update(user_input)
            return await self.async_step_model()

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(step_schema), options
            ),
        )

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Model + generation parameters step."""
        options = self._options
        entry = self._get_entry()
        model_options = await _list_chat_models(self.hass, entry.data)
        current_model = options.get(CONF_CHAT_MODEL, DEFAULT_MODEL)
        if current_model not in model_options:
            model_options = [current_model, *model_options]

        step_schema: VolDictType = {
            vol.Required(
                CONF_CHAT_MODEL,
                default=options.get(CONF_CHAT_MODEL, DEFAULT_MODEL),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[SelectOptionDict(label=m, value=m) for m in model_options],
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=True,
                )
            ),
            vol.Optional(
                CONF_TEMPERATURE,
                default=options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
            ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
            vol.Optional(
                CONF_TOP_P,
                default=options.get(CONF_TOP_P, DEFAULT_TOP_P),
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_MAX_TOKENS,
                default=options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=64, max=8192, step=64, mode=NumberSelectorMode.BOX
                )
            ),
        }

        if user_input is not None:
            options.update(user_input)
            if self._is_new:
                title = options.pop(CONF_NAME, DEFAULT_CONVERSATION_NAME)
                return self.async_create_entry(title=title, data=options)
            return self.async_update_and_abort(
                entry,
                self._get_reconfigure_subentry(),
                data=options,
            )

        return self.async_show_form(
            step_id="model",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(step_schema), options
            ),
        )


class STTSubentryFlow(ConfigSubentryFlow):
    """Add/reconfigure a speech-to-text service under a Scaleway AI entry."""

    def __init__(self) -> None:
        """Init mutable state carried between steps."""
        self._options: dict[str, Any] = {}

    @property
    def _is_new(self) -> bool:
        """True when this is a fresh subentry (vs. a reconfigure)."""
        return self.source == "user"

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Entry point for creating a new STT subentry."""
        self._options = {**RECOMMENDED_STT_OPTIONS}
        return await self.async_step_init()

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Entry point for editing an existing STT subentry."""
        self._options = dict(self._get_reconfigure_subentry().data)
        return await self.async_step_init()

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Configure name, model and optional transcription prompt."""
        entry = self._get_entry()
        if entry.state is not ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        options = self._options
        model_options = await _list_stt_models(self.hass, entry.data)
        current_model = options.get(CONF_STT_MODEL, DEFAULT_STT_MODEL)
        if current_model not in model_options:
            model_options = [current_model, *model_options]

        step_schema: VolDictType = {
            vol.Optional(
                CONF_STT_MODEL,
                default=options.get(CONF_STT_MODEL, DEFAULT_STT_MODEL),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(label=m, value=m) for m in model_options
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=True,
                )
            ),
            vol.Optional(
                CONF_PROMPT,
                description={
                    "suggested_value": options.get(CONF_PROMPT, DEFAULT_STT_PROMPT)
                },
            ): TextSelector(
                TextSelectorConfig(multiline=True, type=TextSelectorType.TEXT)
            ),
        }

        if self._is_new:
            step_schema[
                vol.Required(CONF_NAME, default=DEFAULT_STT_NAME)
            ] = str

        if user_input is not None:
            options.update(user_input)
            if self._is_new:
                title = options.pop(CONF_NAME, DEFAULT_STT_NAME)
                return self.async_create_entry(title=title, data=options)
            return self.async_update_and_abort(
                entry,
                self._get_reconfigure_subentry(),
                data=options,
            )

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(step_schema), options
            ),
        )
