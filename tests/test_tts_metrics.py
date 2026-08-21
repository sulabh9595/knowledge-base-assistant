# Creator: Sulabh Bansod
# Description: Test suite for TTS quality evaluation metrics, SNR, clipping, WER/CER, prosody, and latency profiling.

import pytest
import numpy as np
from app.services.audio_validation_service import AudioValidationService
from app.models.schemas import TTSQualityMetrics


def test_audio_validation_wer_cer_similarity():
    """Verify Word Error Rate, Character Error Rate, and Similarity metrics."""
    validator = AudioValidationService()

    ref = "The quick brown fox jumps over the lazy dog"
    hyp_exact = "The quick brown fox jumps over the lazy dog"
    hyp_err = "The quick brown fox jumped over lazy dog"

    assert validator.word_error_rate(ref, hyp_exact) == 0.0
    assert validator.character_error_rate(ref, hyp_exact) == 0.0
    assert validator.similarity(ref, hyp_exact) == 1.0

    wer_val = validator.word_error_rate(ref, hyp_err)
    cer_val = validator.character_error_rate(ref, hyp_err)
    sim_val = validator.similarity(ref, hyp_err)

    assert 0.0 < wer_val < 0.5
    assert 0.0 < cer_val < 0.5
    assert sim_val > 0.80


def test_audio_signal_metrics_clean():
    """Verify signal-to-noise ratio and clipping detection for clean audio bytes."""
    validator = AudioValidationService()

    # Generate synthetic 1-second 440Hz sine wave (unclipped)
    sr = 24000
    t = np.linspace(0, 1, sr, endpoint=False)
    sine_wave = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    audio_bytes = sine_wave.tobytes()

    sig = validator.calculate_signal_metrics(audio_bytes)
    assert sig["clipping_ratio"] == 0.0
    assert sig["peak_amplitude"] <= 0.6
    assert sig["audio_duration_sec"] >= 0.9
    assert sig["snr_db"] > 0.0


def test_audio_signal_metrics_clipped():
    """Verify clipping detection identifies saturated samples."""
    validator = AudioValidationService()

    # Generate saturated/clipped signal
    sr = 24000
    t = np.linspace(0, 1, sr, endpoint=False)
    clipped_wave = (1.5 * np.sin(2 * np.pi * 440 * t) * 32767).clip(-32767, 32767).astype(np.int16)
    audio_bytes = clipped_wave.tobytes()

    sig = validator.calculate_signal_metrics(audio_bytes)
    assert sig["clipping_ratio"] > 0.0
    assert sig["peak_amplitude"] >= 0.95


def test_prosody_and_wpm_calculation():
    """Verify prosody metrics calculation (words per minute and pause ratio)."""
    validator = AudioValidationService()
    text = "Welcome to the knowledge base assistant voice synthesizer."

    sr = 24000
    t = np.linspace(0, 2, sr * 2, endpoint=False)
    wave = (0.3 * np.sin(2 * np.pi * 300 * t) * 32767).astype(np.int16)

    prosody = validator.calculate_prosody_metrics(wave.tobytes(), text)
    assert prosody["words_per_minute"] > 0.0
    assert "pause_ratio" in prosody


def test_full_tts_quality_evaluation():
    """Verify full multi-dimensional quality metric evaluation."""
    validator = AudioValidationService()
    prompt = "Test full TTS quality metrics evaluation suite."

    sr = 24000
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    wave = (0.4 * np.sin(2 * np.pi * 400 * t) * 32767).astype(np.int16)
    audio_bytes = wave.tobytes()

    metrics = validator.evaluate_full_tts_quality(
        prompt_text=prompt,
        audio_bytes=audio_bytes,
        stt_text=prompt,
        synthesis_time_ms=150.0,
    )

    assert isinstance(metrics, TTSQualityMetrics)
    assert metrics.synthesis_time_ms == 150.0
    assert metrics.word_error_rate == 0.0
    assert metrics.quality_score > 0.0
    assert metrics.overall_quality_pass is True
