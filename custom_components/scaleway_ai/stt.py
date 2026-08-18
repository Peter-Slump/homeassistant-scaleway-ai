"""Speech-to-text platform for Scaleway AI.

Wraps Scaleway's OpenAI-compatible `/v1/audio/transcriptions` endpoint
(`whisper-large-v3` by default) for use in Home Assistant's Assist pipeline.
"""

from __future__ import annotations

from collections.abc import AsyncIterable
import io
import logging
from typing import TYPE_CHECKING, override
import wave

from homeassistant.components import stt
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_PROMPT
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from openai import OpenAIError

from .const import (
    CONF_STT_MODEL,
    DEFAULT_STT_MODEL,
    DEFAULT_STT_PROMPT,
    DOMAIN,
    SUBENTRY_TYPE_STT,
)

if TYPE_CHECKING:
    from . import ScalewayAIConfigEntry

_LOGGER = logging.getLogger(__name__)

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ScalewayAIConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Register one STT entity per speech-to-text subentry."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_TYPE_STT:
            continue
        async_add_entities(
            [ScalewayAISTTEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class ScalewayAISTTEntity(stt.SpeechToTextEntity, Entity):
    """Scaleway Generative APIs speech-to-text entity."""

    _attr_has_entity_name = True
    _attr_name: str | None = None

    def __init__(
        self,
        entry: ScalewayAIConfigEntry,
        subentry: ConfigSubentry,
    ) -> None:
        """Initialise from the parent entry + this STT subentry."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="Scaleway",
            model=subentry.data.get(CONF_STT_MODEL, DEFAULT_STT_MODEL),
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @property
    @override
    def supported_languages(self) -> list[str]:
        """Return languages supported by Whisper on Scaleway."""
        # Whisper large v3 covers ~90 languages; list mirrors OpenAI integration.
        return [
            "af-ZA",
            "ar-SA",
            "hy-AM",
            "az-AZ",
            "be-BY",
            "bs-BA",
            "bg-BG",
            "ca-ES",
            "zh-CN",
            "hr-HR",
            "cs-CZ",
            "da-DK",
            "nl-NL",
            "en-US",
            "et-EE",
            "fi-FI",
            "fr-FR",
            "gl-ES",
            "de-DE",
            "el-GR",
            "he-IL",
            "hi-IN",
            "hu-HU",
            "is-IS",
            "id-ID",
            "it-IT",
            "ja-JP",
            "kn-IN",
            "kk-KZ",
            "ko-KR",
            "lv-LV",
            "lt-LT",
            "mk-MK",
            "ms-MY",
            "mr-IN",
            "mi-NZ",
            "ne-NP",
            "no-NO",
            "fa-IR",
            "pl-PL",
            "pt-PT",
            "ro-RO",
            "ru-RU",
            "sr-RS",
            "sk-SK",
            "sl-SI",
            "es-ES",
            "sw-KE",
            "sv-SE",
            "fil-PH",
            "ta-IN",
            "th-TH",
            "tr-TR",
            "uk-UA",
            "ur-PK",
            "vi-VN",
            "cy-GB",
        ]

    @property
    @override
    def supported_formats(self) -> list[stt.AudioFormats]:
        """Return supported container formats."""
        return [stt.AudioFormats.WAV, stt.AudioFormats.OGG]

    @property
    @override
    def supported_codecs(self) -> list[stt.AudioCodecs]:
        """Return supported audio codecs."""
        return [stt.AudioCodecs.PCM, stt.AudioCodecs.OPUS]

    @property
    @override
    def supported_bit_rates(self) -> list[stt.AudioBitRates]:
        """Return supported bit rates."""
        return [
            stt.AudioBitRates.BITRATE_8,
            stt.AudioBitRates.BITRATE_16,
            stt.AudioBitRates.BITRATE_24,
            stt.AudioBitRates.BITRATE_32,
        ]

    @property
    @override
    def supported_sample_rates(self) -> list[stt.AudioSampleRates]:
        """Return supported sample rates."""
        return [
            stt.AudioSampleRates.SAMPLERATE_8000,
            stt.AudioSampleRates.SAMPLERATE_11000,
            stt.AudioSampleRates.SAMPLERATE_16000,
            stt.AudioSampleRates.SAMPLERATE_18900,
            stt.AudioSampleRates.SAMPLERATE_22000,
            stt.AudioSampleRates.SAMPLERATE_32000,
            stt.AudioSampleRates.SAMPLERATE_37800,
            stt.AudioSampleRates.SAMPLERATE_44100,
            stt.AudioSampleRates.SAMPLERATE_48000,
        ]

    @property
    @override
    def supported_channels(self) -> list[stt.AudioChannels]:
        """Return supported channel layouts."""
        return [stt.AudioChannels.CHANNEL_MONO, stt.AudioChannels.CHANNEL_STEREO]

    @override
    async def async_process_audio_stream(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        """Transcribe an audio stream via Scaleway's Whisper endpoint."""
        audio_bytes = bytearray()
        async for chunk in stream:
            audio_bytes.extend(chunk)
        audio_data = bytes(audio_bytes)

        if metadata.format == stt.AudioFormats.WAV:
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(metadata.channel.value)
                wf.setsampwidth(metadata.bit_rate.value // 8)
                wf.setframerate(metadata.sample_rate.value)
                wf.writeframes(audio_data)
            audio_data = wav_buffer.getvalue()

        options = self.subentry.data
        client = self.entry.runtime_data
        model = options.get(CONF_STT_MODEL, DEFAULT_STT_MODEL)
        prompt = options.get(CONF_PROMPT, DEFAULT_STT_PROMPT)

        try:
            response = await client.audio.transcriptions.create(
                model=model,
                file=(f"audio.{metadata.format.value}", audio_data),
                response_format="json",
                language=metadata.language.split("-")[0],
                prompt=prompt,
            )
        except OpenAIError:
            _LOGGER.exception("Scaleway STT request failed")
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

        if response.text:
            return stt.SpeechResult(response.text, stt.SpeechResultState.SUCCESS)

        return stt.SpeechResult(None, stt.SpeechResultState.ERROR)
