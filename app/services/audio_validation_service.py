"""Helpers for validating TTS/STT audio pipeline behavior."""

import re
from difflib import SequenceMatcher
from typing import Any, Dict, Optional


class AudioValidationService:
    """Validate audio generation and transcription quality."""

    def __init__(self, min_audio_bytes: int = 1):
        self.min_audio_bytes = min_audio_bytes

    def normalize_text(self, text: Optional[str]) -> str:
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def similarity(self, left: Optional[str], right: Optional[str]) -> float:
        left_norm = self.normalize_text(left)
        right_norm = self.normalize_text(right)
        if not left_norm and not right_norm:
            return 1.0
        return round(SequenceMatcher(None, left_norm, right_norm).ratio(), 3)

    def word_error_rate(self, reference: Optional[str], hypothesis: Optional[str]) -> float:
        ref_tokens = self.normalize_text(reference).split()
        hyp_tokens = self.normalize_text(hypothesis).split()

        if not ref_tokens and not hyp_tokens:
            return 0.0
        if not ref_tokens:
            return 1.0
        if not hyp_tokens:
            return 1.0

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

    def validate_tts_output(self, audio_bytes: bytes) -> Dict[str, Any]:
        if not audio_bytes:
            return {
                "ok": False,
                "reason": "empty_audio_bytes",
                "size_bytes": 0,
            }

        return {
            "ok": len(audio_bytes) >= self.min_audio_bytes,
            "reason": "audio_bytes_ok" if len(audio_bytes) >= self.min_audio_bytes else "audio_too_small",
            "size_bytes": len(audio_bytes),
        }

    def validate_stt_output(self, stt_result: Optional[Dict[str, Any]], expected_text: Optional[str]) -> Dict[str, Any]:
        text = (stt_result or {}).get("text", "") if isinstance(stt_result, dict) else ""
        similarity_score = self.similarity(text, expected_text)
        wer = self.word_error_rate(expected_text, text)

        return {
            "ok": similarity_score >= 0.8 or wer <= 0.3,
            "text": text,
            "expected_text": expected_text or "",
            "similarity": similarity_score,
            "word_error_rate": wer,
        }

    def validate_round_trip(
        self,
        expected_text: Optional[str],
        tts_bytes: bytes,
        stt_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        tts_result = self.validate_tts_output(tts_bytes)
        stt_result_data = self.validate_stt_output(stt_result, expected_text)

        return {
            "tts_ok": tts_result["ok"],
            "stt_ok": stt_result_data["ok"],
            "similarity": stt_result_data["similarity"],
            "word_error_rate": stt_result_data["word_error_rate"],
            "tts_details": tts_result,
            "stt_details": stt_result_data,
        }
