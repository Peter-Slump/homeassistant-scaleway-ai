"""Tests for the Scaleway AI STT platform."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock

from homeassistant.components import stt
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_PROMPT
from homeassistant.core import HomeAssistant
import openai
import pytest

from custom_components.scaleway_ai.const import (
    CONF_STT_MODEL,
    DEFAULT_STT_MODEL,
    DEFAULT_STT_PROMPT,
    SUBENTRY_TYPE_STT,
)
from custom_components.scaleway_ai.stt import ScalewayAISTTEntity


async def _audio_stream(data: bytes) -> AsyncIterator[bytes]:
    yield data


def _build_entity(mock_client: MagicMock) -> ScalewayAISTTEntity:
    """Build an STT entity backed by a mocked OpenAI client."""
    entry = MagicMock()
    entry.runtime_data = mock_client
    subentry = ConfigSubentry(
        data=MappingProxyType(
            {
                CONF_STT_MODEL: DEFAULT_STT_MODEL,
                CONF_PROMPT: DEFAULT_STT_PROMPT,
            }
        ),
        subentry_type=SUBENTRY_TYPE_STT,
        title="Scaleway AI STT",
        unique_id="stt-sub",
    )
    entity = ScalewayAISTTEntity(entry, subentry)
    entity.entity_id = "stt.scaleway_ai_stt"
    return entity


@pytest.mark.asyncio
async def test_stt_transcribes_audio(hass: HomeAssistant) -> None:
    """STT entity sends audio to Scaleway and returns the transcript."""
    mock_client = MagicMock()
    mock_transcription = MagicMock()
    mock_transcription.text = "Zet de lampen aan"
    mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_transcription)

    entity = _build_entity(mock_client)
    entity.hass = hass

    metadata = stt.SpeechMetadata(
        language="nl-NL",
        format=stt.AudioFormats.WAV,
        codec=stt.AudioCodecs.PCM,
        bit_rate=stt.AudioBitRates.BITRATE_16,
        sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
        channel=stt.AudioChannels.CHANNEL_MONO,
    )

    result = await entity.async_process_audio_stream(
        metadata, _audio_stream(b"\x00\x01\x02\x03")
    )

    assert result.text == "Zet de lampen aan"
    assert result.result == stt.SpeechResultState.SUCCESS
    mock_client.audio.transcriptions.create.assert_awaited_once()
    call_kwargs = mock_client.audio.transcriptions.create.await_args.kwargs
    assert call_kwargs["model"] == DEFAULT_STT_MODEL
    assert call_kwargs["language"] == "nl"
    assert call_kwargs["prompt"] == DEFAULT_STT_PROMPT


@pytest.mark.asyncio
async def test_stt_api_error_returns_error_state(hass: HomeAssistant) -> None:
    """Scaleway API errors surface as SpeechResultState.ERROR."""
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create = AsyncMock(
        side_effect=openai.APIError(
            "boom",
            request=MagicMock(),
            body=None,
        )
    )

    entity = _build_entity(mock_client)
    entity.hass = hass

    metadata = stt.SpeechMetadata(
        language="en-US",
        format=stt.AudioFormats.OGG,
        codec=stt.AudioCodecs.OPUS,
        bit_rate=stt.AudioBitRates.BITRATE_16,
        sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
        channel=stt.AudioChannels.CHANNEL_MONO,
    )

    result = await entity.async_process_audio_stream(
        metadata, _audio_stream(b"fake-opus")
    )

    assert result.text is None
    assert result.result == stt.SpeechResultState.ERROR
