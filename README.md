# Scaleway AI for Home Assistant

A Home Assistant custom integration that connects Home Assistant's Assist pipeline
to [Scaleway's Generative APIs](https://www.scaleway.com/en/generative-apis/) —
EU-hosted, OpenAI-compatible LLM inference running in `fr-par`.

> **Status:** v0.2 ships a conversation agent and speech-to-text via Whisper.
> Text-to-speech is not offered by Scaleway as a hosted service today and will
> only be added if that changes or via user-supplied Managed Inference endpoints.

## Features

- **Conversation agent** — plug Scaleway's chat models (Mistral, Llama, Qwen,
  GPT-OSS, DeepSeek, GLM, …) into Home Assistant's built-in Assist pipeline.
- **Speech-to-text** — transcribe voice input via Scaleway's `whisper-large-v3`
  model (or other listed Whisper/Voxtral models) for full voice Assist pipelines.
- **Multi-persona** — create as many conversation agents and STT engines as you
  like from a single Scaleway credential, each with its own model, prompt and
  settings.
- **HA tool calling** — agents can control devices, read sensor states and run
  scripts via Home Assistant's `assist` LLM API.
- **Streaming responses** — token deltas surface in the Assist UI as they
  arrive.
- **EU-native** — inference stays in `fr-par` (France). No data leaves the EU.
- **Bring your own endpoint** — advanced users can point the integration at a
  Scaleway Managed Inference dedicated deployment (`https://<uuid>.ifr.fr-par.scaleway.com/v1/`)
  or any other OpenAI-compatible base URL.

## Installation

### Via HACS (recommended)

1. In HACS → Integrations → ⋮ → *Custom repositories*, add
   `https://github.com/Peter-Slump/homeassistant-scaleway-ai` as an
   *Integration*.
2. Install **Scaleway AI**.
3. Restart Home Assistant.
4. Settings → Devices & services → *Add integration* → **Scaleway AI**.

### Manual

Copy `custom_components/scaleway_ai/` into your Home Assistant `config/custom_components/`
directory and restart.

## Configuration

1. Get a Scaleway **API secret key** (IAM → API Keys → *Generate new key*, make
   sure it has the *Generative APIs Full Access* permission).
2. Add the integration in Home Assistant. You'll be asked for:
   - **API key** — the secret key from step 1.
   - **Project ID** *(optional)* — scopes calls to a specific Scaleway project.
   - **Base URL** *(advanced)* — defaults to `https://api.scaleway.ai/v1`. Set
     this to your Managed Inference endpoint URL to use a dedicated deployment.
3. On first setup, default **Conversation agent** and **Speech-to-text**
   subentries are created. Add more subentries from the integration's configure
   menu if you want separate personas or STT profiles.
4. In Settings → Voice assistants, create an Assist pipeline:
   - **Speech-to-text:** Scaleway AI STT
   - **Conversation agent:** Scaleway AI Conversation
   - **Text-to-speech:** e.g. Piper or another local/cloud TTS engine

## Roadmap

- v0.1 — Conversation agent ✅
- v0.2 — Speech-to-text via `whisper-large-v3` ✅
- v0.x — Text-to-speech (only if/when Scaleway ships a hosted TTS model, or via
  user-supplied Managed Inference URL)

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_test.txt
pip install pre-commit ruff mypy
pre-commit install
pytest
```

CI runs `hassfest`, HACS validation, `ruff`, `mypy` and `pytest` on every
push and pull request.

## License

MIT — see [LICENSE](LICENSE).
