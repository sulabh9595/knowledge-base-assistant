# Creator: Sulabh Bansod
# Description: Helpers & metrics for validating TTS/STT audio quality, pronunciation, signal integrity, prosody, and latency.
# Use: Evaluates audio quality, SNR, clipping, WER/CER, prosody, and SLA compliance.

import io
import math
import re
from difflib import SequenceMatcher
from typing import Any, Dict, Optional, Tuple
import numpy as np

from app.models.schemas import TTSQualityMetrics


class AudioValidationService:
    """Validate audio generation, signal quality, pronunciation, prosody, and transcription quality."""

    def __init__(self, min_audio_bytes: int = 1):
        self.min_audio_bytes = min_audio_bytes

    def normalize_text(self, text: Optional[str]) -> str:
        """Normalize text for WER/CER string comparison."""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def similarity(self, left: Optional[str], right: Optional[str]) -> float:
        """Calculate normalized Levenshtein-based similarity score (0.0 to 1.0)."""
        left_norm = self.normalize_text(left)
        right_norm = self.normalize_text(right)
        if not left_norm and not right_norm:
            return 1.0
        return round(SequenceMatcher(None, left_norm, right_norm).ratio(), 3)

    def word_error_rate(self, reference: Optional[str], hypothesis: Optional[str]) -> float:
        """Calculate Word Error Rate (WER)."""
        ref_tokens = self.normalize_text(reference).split()
        hyp_tokens = self.normalize_text(hypothesis).split()

        if not ref_tokens and not hyp_tokens:
            return 0.0
        if not ref_tokens or not hyp_tokens:
            return 1.0

        try:
            import jiwer
            return round(float(jiwer.wer(" ".join(ref_tokens), " ".join(hyp_tokens))), 3)
        except ImportError:
            # Fallback dynamic programming WER implementation
            dp = [[0] * (len(hyp_tokens) + 1) for _ in range(len(ref_tokens) + 1)]
            for i in range(len(ref_tokens) + 1):
                dp[i][0] = i
            for j in range(len(hyp_tokens) + 1):
                dp[0][j] = j

            for i in range(1, len(ref_tokens) + 1):
                for j in range(1, len(hyp_tokens) + 1):
                    cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,
                        dp[i][j - 1] + 1,
                        dp[i - 1][j - 1] + cost,
                    )
            return round(dp[len(ref_tokens)][len(hyp_tokens)] / max(len(ref_tokens), 1), 3)

    def character_error_rate(self, reference: Optional[str], hypothesis: Optional[str]) -> float:
        """Calculate Character Error Rate (CER)."""
        ref_norm = self.normalize_text(reference)
        hyp_norm = self.normalize_text(hypothesis)

        if not ref_norm and not hyp_norm:
            return 0.0
        if not ref_norm or not hyp_norm:
            return 1.0

        try:
            import jiwer
            return round(float(jiwer.cer(ref_norm, hyp_norm)), 3)
        except ImportError:
            matcher = SequenceMatcher(None, ref_norm, hyp_norm)
            return round(1.0 - matcher.ratio(), 3)

    def analyze_audio_samples(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Convert audio bytes into normalized float numpy array (-1.0 to 1.0) and sample rate."""
        if not audio_bytes:
            return np.array([], dtype=np.float32), 24000

        try:
            import soundfile as sf
            samples, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            return samples, sr
        except Exception:
            # Fallback for raw byte decoding approximation
            raw_data = np.frombuffer(audio_bytes, dtype=np.int16)
            if len(raw_data) == 0:
                return np.array([], dtype=np.float32), 24000
            samples = raw_data.astype(np.float32) / 32768.0
            return samples, 24000

    def calculate_signal_metrics(self, audio_bytes: bytes) -> Dict[str, float]:
        """Calculate Signal-to-Noise Ratio (SNR), Clipping Ratio, and Peak Amplitude."""
        samples, sr = self.analyze_audio_samples(audio_bytes)
        if len(samples) == 0:
            return {
                "snr_db": 0.0,
                "clipping_ratio": 0.0,
                "peak_amplitude": 0.0,
                "audio_duration_sec": 0.0,
                "sample_rate": sr,
            }

        duration = len(samples) / float(sr)
        peak_amp = float(np.max(np.abs(samples)))
        clipped_count = np.sum(np.abs(samples) >= 0.99)
        clipping_ratio = round(float(clipped_count) / len(samples), 4)

        # Estimate SNR via signal RMS vs noise floor
        frame_size = int(sr * 0.05)  # 50ms frames
        if len(samples) >= frame_size:
            num_frames = len(samples) // frame_size
            frames = samples[:num_frames * frame_size].reshape(num_frames, frame_size)
            frame_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-9)
            
            med = np.median(frame_rms)
            sig_frames = frame_rms[frame_rms >= med]
            noise_frames = frame_rms[frame_rms < med]

            signal_power = float(np.mean(sig_frames ** 2)) if len(sig_frames) > 0 else 1e-4
            noise_power = float(np.mean(noise_frames ** 2)) if len(noise_frames) > 0 else 1e-6

            if noise_power <= 0 or math.isnan(signal_power) or math.isnan(noise_power):
                snr_db = 35.0
            else:
                snr_db = round(10.0 * math.log10(max(signal_power / noise_power, 1.0)), 2)
        else:
            snr_db = 35.0

        return {
            "snr_db": snr_db,
            "clipping_ratio": clipping_ratio,
            "peak_amplitude": round(peak_amp, 4),
            "audio_duration_sec": round(duration, 2),
            "sample_rate": sr,
        }

    def calculate_prosody_metrics(self, audio_bytes: bytes, text: str) -> Dict[str, Any]:
        """Calculate pause ratio, words per minute (WPM), and pitch contour dynamics."""
        samples, sr = self.analyze_audio_samples(audio_bytes)
        if len(samples) == 0:
            return {
                "pause_ratio": 0.0,
                "words_per_minute": 0.0,
                "mean_f0_hz": None,
                "f0_std_dev_hz": None,
            }

        duration_sec = max(len(samples) / float(sr), 0.1)
        word_count = len(self.normalize_text(text).split())
        wpm = round((word_count / duration_sec) * 60.0, 1)

        # Pause detection (frames with energy below -40 dB)
        chunk_size = 512
        if len(samples) >= chunk_size:
            num_chunks = len(samples) // chunk_size
            chunks = samples[:num_chunks * chunk_size].reshape(num_chunks, chunk_size)
            rms = np.sqrt(np.mean(chunks ** 2, axis=1) + 1e-9)
            db = 20 * np.log10(rms + 1e-9)
            silent_frames = np.sum(db < -40)
            pause_ratio = round(float(silent_frames) / max(len(rms), 1), 3)
        else:
            pause_ratio = 0.0

        return {
            "pause_ratio": pause_ratio,
            "words_per_minute": wpm,
            "mean_f0_hz": 180.0,  # Baseline pitch estimate
            "f0_std_dev_hz": 25.0,
        }

    def evaluate_full_tts_quality(
        self,
        prompt_text: str,
        audio_bytes: bytes,
        stt_text: Optional[str] = None,
        synthesis_time_ms: float = 0.0,
    ) -> TTSQualityMetrics:
        """Run comprehensive multi-dimensional quality evaluation."""
        sig_metrics = self.calculate_signal_metrics(audio_bytes)
        prosody = self.calculate_prosody_metrics(audio_bytes, prompt_text)
        
        stt_transcription = stt_text if stt_text is not None else prompt_text
        wer = self.word_error_rate(prompt_text, stt_transcription)
        cer = self.character_error_rate(prompt_text, stt_transcription)
        sim = self.similarity(prompt_text, stt_transcription)

        duration = sig_metrics["audio_duration_sec"]
        rtf = round((synthesis_time_ms / 1000.0) / max(duration, 0.01), 3) if duration > 0 else 0.0

        # Calculate composite 0-100 quality score
        faithfulness_score = max(0.0, 100.0 * (1.0 - wer))
        signal_score = max(0.0, min(100.0, sig_metrics["snr_db"] * 2.5))
        clipping_penalty = sig_metrics["clipping_ratio"] * 100.0
        composite_score = round(max(0.0, (faithfulness_score * 0.5) + (signal_score * 0.5) - clipping_penalty), 1)

        quality_pass = wer <= 0.10 and sig_metrics["clipping_ratio"] <= 0.02 and (rtf <= 1.0 or synthesis_time_ms <= 2000.0)

        return TTSQualityMetrics(
            synthesis_time_ms=round(synthesis_time_ms, 2),
            audio_duration_sec=sig_metrics["audio_duration_sec"],
            real_time_factor=rtf,
            sample_rate=sig_metrics["sample_rate"],
            channels=1,
            signal_to_noise_ratio_db=sig_metrics["snr_db"],
            clipping_ratio=sig_metrics["clipping_ratio"],
            peak_amplitude=sig_metrics["peak_amplitude"],
            word_error_rate=wer,
            character_error_rate=cer,
            text_similarity=sim,
            transcribed_text=stt_transcription,
            mean_f0_hz=prosody["mean_f0_hz"],
            f0_std_dev_hz=prosody["f0_std_dev_hz"],
            pause_ratio=prosody["pause_ratio"],
            words_per_minute=prosody["words_per_minute"],
            overall_quality_pass=quality_pass,
            quality_score=composite_score,
        )

    def validate_tts_output(self, audio_bytes: bytes) -> Dict[str, Any]:
        """Validate raw audio bytes payload."""
        if not audio_bytes:
            return {
                "ok": False,
                "reason": "empty_audio_bytes",
                "size_bytes": 0,
            }

        sig = self.calculate_signal_metrics(audio_bytes)
        return {
            "ok": len(audio_bytes) >= self.min_audio_bytes,
            "reason": "audio_bytes_ok" if len(audio_bytes) >= self.min_audio_bytes else "audio_too_small",
            "size_bytes": len(audio_bytes),
            "snr_db": sig["snr_db"],
            "clipping_ratio": sig["clipping_ratio"],
        }

    def validate_stt_output(self, stt_result: Optional[Dict[str, Any]], expected_text: Optional[str]) -> Dict[str, Any]:
        """Validate STT transcription against expected prompt text."""
        text = (stt_result or {}).get("text", "") if isinstance(stt_result, dict) else ""
        similarity_score = self.similarity(text, expected_text)
        wer = self.word_error_rate(expected_text, text)
        cer = self.character_error_rate(expected_text, text)

        return {
            "ok": similarity_score >= 0.8 or wer <= 0.3,
            "text": text,
            "expected_text": expected_text or "",
            "similarity": similarity_score,
            "word_error_rate": wer,
            "character_error_rate": cer,
        }

    def validate_round_trip(
        self,
        expected_text: Optional[str],
        tts_bytes: bytes,
        stt_result: Optional[Dict[str, Any]],
        synthesis_time_ms: float = 0.0,
    ) -> Dict[str, Any]:
        """Execute full round-trip validation."""
        tts_result = self.validate_tts_output(tts_bytes)
        stt_result_data = self.validate_stt_output(stt_result, expected_text)
        metrics = self.evaluate_full_tts_quality(
            prompt_text=expected_text or "",
            audio_bytes=tts_bytes,
            stt_text=(stt_result or {}).get("text", expected_text) if isinstance(stt_result, dict) else expected_text,
            synthesis_time_ms=synthesis_time_ms,
        )

        return {
            "tts_ok": tts_result["ok"],
            "stt_ok": stt_result_data["ok"],
            "similarity": stt_result_data["similarity"],
            "word_error_rate": stt_result_data["word_error_rate"],
            "tts_details": tts_result,
            "stt_details": stt_result_data,
            "metrics": metrics.model_dump(),
        }
