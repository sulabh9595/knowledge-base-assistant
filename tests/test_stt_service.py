# Creator: Sulabh Bansod
# Description: Unit tests for STTService and audio processing logic.

from unittest.mock import MagicMock, patch
import pytest

from app.services.stt_service import STTService
from app.loaders.file_loader import FileLoader


def test_stt_service_empty_bytes():
    service = STTService()
    res = service.transcribe_bytes(b"", filename="empty.wav")
    assert res["text"] == ""
    assert res["duration"] == 0.0
    assert res["segments"] == []


@patch("app.services.stt_service.STTService.model")
def test_stt_service_mock_transcribe(mock_model):
    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 2.5
    mock_segment.text = "Hello world audio test"

    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.98
    mock_info.duration = 2.5

    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    service = STTService()
    res = service.transcribe_bytes(b"dummy_audio_bytes", filename="test.wav")

    assert res["text"] == "Hello world audio test"
    assert res["language"] == "en"
    assert res["duration"] == 2.5
    assert len(res["segments"]) == 1
    assert res["segments"][0]["text"] == "Hello world audio test"


@patch("app.services.stt_service.stt_service.transcribe_bytes")
def test_file_loader_read_audio(mock_transcribe):
    mock_transcribe.return_value = {
        "text": "Audio transcription sample text",
        "language": "en",
        "duration": 5.0,
        "segments": []
    }

    result = FileLoader.read_audio(b"audio_bytes", filename="sample.mp3")
    assert result["text"] == "Audio transcription sample text"
    assert result["duration"] == 5.0
