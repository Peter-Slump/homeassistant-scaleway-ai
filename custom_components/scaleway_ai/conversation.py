"""Conversation platform for Scaleway AI.

Thin wrapper: one `ScalewayAIConversationEntity` per `conversation`
subentry of a parent Scaleway AI config entry. All request/response
plumbing lives in `entity.ScalewayAIBaseLLMEntity._async_handle_chat_log`.
"""

from __future__ import annotations

from typing import Literal

from homeassistant.components import conversation
from homeassistant.const import CONF_LLM_HASS_API, CONF_PROMPT, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import ScalewayAIConfigEntry
from .const import DOMAIN, SUBENTRY_TYPE_CONVERSATION
from .entity import ScalewayAIBaseLLMEntity

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ScalewayAIConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Register one conversation entity per conversation subentry."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_TYPE_CONVERSATION:
            continue
        async_add_entities(
            [ScalewayAIConversationEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class ScalewayAIConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,  # type: ignore[name-defined,misc]
    ScalewayAIBaseLLMEntity,
):
    """Conversation agent backed by Scaleway's Generative APIs."""

    _attr_supports_streaming = True

    def __init__(self, entry: ScalewayAIConfigEntry, subentry) -> None:  # type: ignore[no-untyped-def]
        """Initialise from the parent entry + this conversation subentry."""
        super().__init__(entry, subentry)
        if self.subentry.data.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Report supported languages. Scaleway models are broadly multilingual."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """Register this entity as the active conversation agent for its entry."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """Unregister when removed."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Handle one user turn: wire up LLM API + prompt, then run the tool loop."""
        options = self.subentry.data
        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        await self._async_handle_chat_log(chat_log)
        return conversation.async_get_result_from_chat_log(user_input, chat_log)
