# Creator: Sulabh Bansod
# Description: Unit test suite for KokoroTTSService local neural TTS.

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from app.services.kokoro_tts_service import KokoroTTSService


def test_kokoro_tts_empty_text():
    """Verify empty or whitespace text returns empty bytes and string."""
    service = KokoroTTSService()
    assert service.synthesize_bytes("") == b""
    assert service.synthesize_bytes("   ") == b""
    assert service.synthesize_base64("") == ""


@patch("app.services.kokoro_tts_service.KokoroTTSService._get_pipeline")
def test_kokoro_tts_synthesize_bytes_mock(mock_get_pipeline):
    """Verify synthesize_bytes produces WAV bytes with mocked Kokoro pipeline."""
    mock_pipeline = MagicMock()
    dummy_audio = np.zeros(24000, dtype=np.float32)
    mock_pipeline.return_value = [("gs", "ps", dummy_audio)]
    mock_get_pipeline.return_value = mock_pipeline

    mock_sf = MagicMock()
    def fake_sf_write(file_obj, data, samplerate, format):
        file_obj.write(b"RIFF_FAKE_WAV_HEADER_AND_DATA")
    mock_sf.write.side_effect = fake_sf_write

    with patch.dict("sys.modules", {"soundfile": mock_sf}):
        service = KokoroTTSService(default_voice="af_heart")
        audio_bytes = service.synthesize_bytes("Hello world from Kokoro TTS!", voice="af_heart")

        assert audio_bytes == b"RIFF_FAKE_WAV_HEADER_AND_DATA"
        mock_pipeline.assert_called_once_with(
            "Hello world from Kokoro TTS!",
            voice="af_heart",
            speed=1.0,
            split_pattern=r'\n+'
        )


@patch("app.services.kokoro_tts_service.KokoroTTSService.synthesize_bytes")
def test_kokoro_tts_synthesize_base64_mock(mock_synth_bytes):
    """Verify synthesize_base64 returns base64 string from audio bytes."""
    mock_synth_bytes.return_value = b"FAKE_AUDIO_BYTES"
    service = KokoroTTSService()
    b64_out = service.synthesize_base64("Testing base64 encoding")

    assert isinstance(b64_out, str)
    assert len(b64_out) > 0
