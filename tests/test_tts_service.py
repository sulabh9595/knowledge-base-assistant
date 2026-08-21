"""Unit test suite for TTSService audio synthesis."""

from typing import Optional

import pytest
from unittest.mock import patch, AsyncMock
from app.services.tts_service import TTSService, tts_service


@pytest.mark.asyncio
async def test_tts_service_empty_text():
    """Verify empty or whitespace text returns empty audio bytes and string."""
    bytes_res = await tts_service.synthesize_bytes("")
    assert bytes_res == b""

    b64_res = await tts_service.synthesize_base64("   ")
    assert b64_res == ""


@pytest.mark.asyncio
async def test_tts_service_mock_synthesis():
    """Verify synthesize_bytes and synthesize_base64 work correctly with mock provider."""
    service = TTSService(provider="edge-tts")

    # Mock synthesize_bytes to return fake audio bytes
    with patch.object(service, "synthesize_bytes", new_callable=AsyncMock) as mock_synth:
        mock_synth.return_value = b"FAKE_MP3_AUDIO_DATA"

        b64_output = await service.synthesize_base64("Hello world")

        assert mock_synth.called
        assert b64_output != ""
        import base64
        decoded = base64.b64decode(b64_output)
        assert decoded == b"FAKE_MP3_AUDIO_DATA"


@pytest.mark.asyncio
async def test_tts_service_fallback_on_error():
    """Verify fallback mechanism when primary engine fails."""
    service = TTSService(provider="non_existent_provider")
    # Calling synthesize_bytes should complete gracefully without unhandled crashes
    result = await service.synthesize_bytes("Test fallback")
    assert isinstance(result, bytes)


@pytest.mark.asyncio
async def test_tts_service_azure_provider_mock():
    """Verify Azure provider branch is invoked and returns audio bytes."""
    service = TTSService(provider="azure")

    class MockAzureSpeechService:
        def synthesize_bytes(self, text: str, voice: Optional[str] = None, audio_format: str = "mp3") -> bytes:
            return b"AZURE_FAKE_AUDIO"

    with patch.object(service, "_create_azure_service", return_value=MockAzureSpeechService()):
        result = await service.synthesize_bytes("Hello from Azure")
        assert result == b"AZURE_FAKE_AUDIO"


@pytest.mark.asyncio
async def test_tts_service_kokoro_provider_mock():
    """Verify Kokoro provider branch is invoked and returns audio bytes."""
    service = TTSService(provider="kokoro")

    class MockKokoroSpeechService:
        def synthesize_bytes(self, text: str, voice: Optional[str] = None, speed: float = 1.0, audio_format: str = "wav") -> bytes:
            return b"KOKORO_FAKE_AUDIO"

    with patch.object(service, "_create_kokoro_service", return_value=MockKokoroSpeechService()):
        result = await service.synthesize_bytes("Hello from Kokoro TTS", voice="af_heart")
        assert result == b"KOKORO_FAKE_AUDIO"
