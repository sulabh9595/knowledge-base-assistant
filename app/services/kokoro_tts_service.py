# Creator: Sulabh Bansod
# Description: Service wrapper for Kokoro-82M Text-to-Speech (TTS) local neural synthesis.
# Use: Provides offline, local neural speech synthesis using Kokoro KPipeline.

import base64
import io
import logging
from typing import Optional
import numpy as np

from app.config.settings import settings

logger = logging.getLogger(__name__)


class KokoroTTSService:
    """Service wrapper for Kokoro local neural text-to-speech engine."""

    def __init__(self, default_voice: Optional[str] = None, default_lang: Optional[str] = None):
        self.default_voice = default_voice or getattr(settings, "kokoro_tts_voice", "af_heart")
        self.default_lang = default_lang or getattr(settings, "kokoro_tts_lang_code", "a")
        self._pipeline = None

    def _get_pipeline(self, lang_code: Optional[str] = None):
        """Lazy-load the Kokoro KPipeline singleton."""
        if self._pipeline is None:
            try:
                from kokoro import KPipeline
                lang = lang_code or self.default_lang
                logger.info(f"Initializing Kokoro TTS KPipeline with lang_code='{lang}'...")
                self._pipeline = KPipeline(lang_code=lang)
            except ImportError:
                logger.error("kokoro package is not installed. Please run `pip install kokoro soundfile`.")
                raise RuntimeError("Kokoro package not installed")
            except Exception as e:
                logger.error(f"Failed to initialize Kokoro TTS pipeline: {e}")
                raise RuntimeError(f"Kokoro TTS initialization error: {e}")
        return self._pipeline

    def synthesize_bytes(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        audio_format: str = "wav"
    ) -> bytes:
        """Synthesize text into raw audio bytes (WAV)."""
        if not text or not text.strip():
            return b""

        try:
            import soundfile as sf
            pipeline = self._get_pipeline()
            selected_voice = voice or self.default_voice

            # Clean text formatting
            clean_text = text.replace("#", "").replace("*", "").replace("`", "").strip()

            generator = pipeline(clean_text, voice=selected_voice, speed=speed, split_pattern=r'\n+')
            audio_segments = []

            for gs, ps, audio in generator:
                if audio is not None:
                    audio_segments.append(audio)

            if not audio_segments:
                logger.warning("Kokoro TTS generated empty audio segments.")
                return b""

            full_audio = np.concatenate(audio_segments)

            buffer = io.BytesIO()
            sf.write(buffer, full_audio, 24000, format='WAV')
            buffer.seek(0)
            return buffer.read()
        except Exception as exc:
            logger.error(f"Kokoro TTS synthesis failed: {exc}")
            raise

    def synthesize_base64(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0
    ) -> str:
        """Synthesize text into Base64-encoded audio string."""
        audio_bytes = self.synthesize_bytes(text, voice=voice, speed=speed)
        if not audio_bytes:
            return ""
        return base64.b64encode(audio_bytes).decode("utf-8")
