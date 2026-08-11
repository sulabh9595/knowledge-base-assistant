from app.services.audio_validation_service import AudioValidationService


def test_validate_tts_output_accepts_non_empty_bytes():
    service = AudioValidationService()
    result = service.validate_tts_output(b"fake-audio-bytes")

    assert result["ok"] is True
    assert result["reason"] == "audio_bytes_ok"
    assert result["size_bytes"] == len(b"fake-audio-bytes")


def test_validate_tts_output_rejects_empty_bytes():
    service = AudioValidationService()
    result = service.validate_tts_output(b"")

    assert result["ok"] is False
    assert "empty" in result["reason"].lower()


def test_validate_stt_output_matches_expected_text():
    service = AudioValidationService()
    result = service.validate_stt_output(
        {"text": "Hello world"},
        "hello world",
    )

    assert result["ok"] is True
    assert result["similarity"] >= 0.95
    assert result["word_error_rate"] == 0.0


def test_validate_round_trip_reports_similarity_and_wer():
    service = AudioValidationService()
    result = service.validate_round_trip(
        expected_text="Hello world",
        tts_bytes=b"fake-audio-bytes",
        stt_result={"text": "hello, world!"},
    )

    assert result["tts_ok"] is True
    assert result["stt_ok"] is True
    assert result["similarity"] >= 0.8
    assert result["word_error_rate"] >= 0.0
