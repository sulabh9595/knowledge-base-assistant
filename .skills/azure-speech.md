# Azure AI Speech Integration Plan

> **Purpose:** Define a detailed implementation plan to integrate Azure AI Speech as a separate speech service from OPIK/DeepEval in the Knowledge Base Application.
> **Scope:** Add Azure-based Speech-to-Text and Text-to-Speech while preserving existing local/edge TTS and the separation from OPIK-based validation.

## Objective

Implement Azure AI Speech to provide enterprise-grade speech recognition and synthesis as a distinct service layer. This should support:
- Azure Speech-to-Text for voice query ingestion and audio document transcription
- Azure Text-to-Speech for high-quality spoken output
- Configuration-driven provider selection so Azure speech can be enabled without changing core business logic
- Independent Azure speech service code paths that do not couple speech execution to OPIK validation

## Why Azure AI Speech?

- enterprise-quality neural speech models
- managed cloud deployment for production reliability
- support for Azure region, key, and custom voice configuration
- better multilingual coverage and voice quality than local fallback providers
- separate service from OPIK so speech is handled as a distinct capability

## Design Principles

- Keep Azure speech separate from OPIK/DeepEval evaluation logic.
- Use a provider abstraction in `app/services/tts_service.py` and `app/services/stt_service.py`.
- Prefer configuration over hardcoding.
- Support fallback to existing local providers when Azure is not configured.
- Add dedicated API endpoints for Azure speech if useful.
- Add tests for Azure speech provider integration.

## Implementation Phases

### Phase 1: Provider Abstraction and Configuration

#### 1.1 Add Azure speech config fields (`app/config/settings.py`)
- `azure_speech_enabled: bool = False`
- `azure_speech_key: Optional[str] = None`
- `azure_speech_region: Optional[str] = None`
- `azure_speech_tts_voice: str = "en-US-JennyNeural"`
- `azure_stt_language: Optional[str] = "en-US"`
- `azure_speech_endpoint: Optional[str] = None`
- `azure_speech_cache_enabled: bool = False`

#### 1.2 Extend provider selection
- `tts_provider` can be `edge-tts`, `azure`, `gTTS`, `say`, `pyttsx3`
- `stt_provider` can be `faster-whisper`, `azure`, `openai-whisper` (future)

#### 1.3 Install Azure SDK dependency
- Add `azure-cognitiveservices-speech` to `requirements.txt` / `pyproject.toml`

### Phase 2: Azure Speech Service Implementation

#### 2.1 `app/services/azure_speech_service.py`
- Add a dedicated wrapper for Azure Speech SDK functionality.
- Implement:
  - `transcribe_bytes(audio_bytes, filename=None, language=None)`
  - `synthesize_bytes(text, voice=None, format='mp3')`
  - `synthesize_base64(text, voice=None, format='mp3')`
- Support both batch and streaming transcription if needed.
- Use platform-friendly `SpeechConfig(subscription=key, region=region)`.
- Use `AudioDataStream` or `AudioOutputConfig` for TTS output bytes.

#### 2.2 `app/services/stt_service.py`
- Add Azure provider branch:
  - if `provider == 'azure'` and Azure configured, call `azure_speech_service.transcribe_bytes`
  - preserve local faster-whisper provider as fallback
- Keep existing `transcribe_bytes` API stable.

#### 2.3 `app/services/tts_service.py`
- Add Azure provider branch:
  - if `provider == 'azure'` and Azure configured, call `azure_speech_service.synthesize_bytes`
  - preserve existing `edge-tts` -> `gTTS` -> `say` -> `pyttsx3` fallback chain
- Allow voice override through request parameters.

### Phase 3: API and UI Integration

#### 3.1 API enhancements
- Update `app/api/tts.py` to support provider selection and Azure-specific voice parameters.
- If desired, add `POST /speech/tts` and `POST /speech/stt` endpoints for explicit Azure speech calls.
- Keep `/rag/query` and `/agent/langgraph/query` audio attachment logic unchanged; they can use configured provider automatically.

#### 3.2 Frontend support
- Add Azure-specific configuration fields to the Streamlit TTS settings section.
- Add a provider dropdown with values: `edge-tts`, `azure`, `gTTS`, `macos-say`, `pyttsx3`.
- Expose Azure voice selection when `azure` provider is chosen.

### Phase 4: Validation and Testing

#### 4.1 Unit tests
- Add tests for Azure STT provider branch in `tests/test_stt_service.py`.
- Add tests for Azure TTS provider branch in `tests/test_tts_service.py`.
- Mock Azure SDK responses so tests do not require live Azure credentials.

#### 4.2 Integration tests
- Add API tests around Azure speech provider selection in `tests/test_tts_api.py` and `tests/test_audio_ingestion_api.py`.
- Validate that when Azure is configured, the application routes audio to Azure and returns valid bytes or base64.

#### 4.3 Documentation
- Update `README.md` with Azure Speech instructions.
- Document environment variables:
  - `AZURE_SPEECH_KEY`
  - `AZURE_SPEECH_REGION`
  - `AZURE_SPEECH_ENDPOINT`
  - `AZURE_TTS_VOICE`
  - `AZURE_STT_LANGUAGE`
- Document that Azure Speech is a separate service from OPIK and is used only for audio input/output, not for evaluation.

### Phase 5: Optional Advanced Enhancements

- Add Azure speech health checks in `app/api/health.py`.
- Add telemetry or logging for Azure speech usage.
- Add optional Azure custom voice and pronunciation support.
- Add offline fallback behavior: if Azure fails, gracefully switch to existing local/audio providers.

## Implementation Notes

- Keep OPIK / DeepEval separate from Azure speech. OPIK remains the evaluation layer and should not depend on Azure speech internals.
- Treat Azure AI Speech as a provider implementation for audio I/O only.
- Use dependency inversion: API controllers depend on `stt_service`/`tts_service`, not directly on Azure SDK.
- Provide clear error messages when Azure credentials are missing or invalid.
- Preserve the current local-first architecture by making Azure optional.

## Recommended File Changes

- `app/config/settings.py`
- `app/services/azure_speech_service.py`
- `app/services/stt_service.py`
- `app/services/tts_service.py`
- `app/api/tts.py`
- `frontend/app.py` (optional provider UI updates)
- `tests/test_stt_service.py`
- `tests/test_tts_service.py`
- `tests/test_tts_api.py`
- `README.md`

## Success Criteria

- Azure Speech integration works as a configurable provider.
- Audio ingestion and TTS can use Azure without affecting OPIK or evaluation logic.
- Existing local audio flow remains intact as a fallback.
- Provider selection is clear to users and maintainers.
- Test coverage includes Azure provider branches and configuration.
