"""Service wrapper for Text-to-Speech (TTS) audio synthesis."""

import base64
import logging
import os
import tempfile
import asyncio
import subprocess
from typing import Dict, Any, Optional, Tuple

from app.config.settings import settings

logger = logging.getLogger(__name__)


KOKORO_VOICE_MAP = {
    "af_heart": "en-US-AvaNeural",
    "af_bella": "en-US-EmmaNeural",
    "am_adam": "en-US-AndrewNeural",
    "am_michael": "en-US-GuyNeural",
    "bf_emma": "en-GB-SoniaNeural",
}


def _resolve_edge_voice(voice: Optional[str]) -> str:
    """Map Kokoro voice names to equivalent high-quality Edge TTS neural voices."""
    if not voice:
        return getattr(settings, "tts_default_voice", "en-US-AvaNeural")
    return KOKORO_VOICE_MAP.get(voice, voice)


class TTSService:
    """Service wrapper for Text-to-Speech (TTS) audio synthesis."""

    def __init__(self, provider: Optional[str] = None, default_voice: Optional[str] = None):
        self.provider = provider or getattr(settings, "tts_provider", "edge-tts")
        self.default_voice = default_voice or getattr(settings, "tts_default_voice", "en-US-AvaNeural")
        self.audio_format = getattr(settings, "tts_audio_format", "mp3")

    async def synthesize_bytes(self, text: str, voice: Optional[str] = None) -> bytes:
        """Synthesize text into raw audio bytes (MP3/WAV)."""
        if not text or not text.strip():
            return b""

        # Clean text snippet (strip markdown formatting headers/bullets for cleaner speech output)
        clean_text = text.replace("#", "").replace("*", "").replace("`", "").strip()
        selected_voice = voice or self.default_voice

        # Kokoro TTS provider (Local Neural TTS)
        if self.provider == "kokoro":
            try:
                kokoro_service = self._create_kokoro_service()
                data = await asyncio.to_thread(kokoro_service.synthesize_bytes, clean_text, voice=selected_voice)
                if data:
                    logger.info("Synthesized audio successfully using Kokoro TTS.")
                    return data
            except Exception as e:
                logger.warning(f"Kokoro TTS synthesis failed: {e}. Falling back to edge-tts/other engines with mapped voice...")

        # Azure Speech provider
        if self.provider == "azure":
            try:
                azure_service = self._create_azure_service()
                data = azure_service.synthesize_bytes(clean_text, voice=selected_voice, audio_format=self.audio_format)
                if data:
                    logger.info("Synthesized audio successfully using Azure Speech.")
                    return data
            except Exception as e:
                logger.warning(f"Azure Speech synthesis failed: {e}. Falling back to existing engines...")

        # 1. Primary / Fallback Engine: edge-tts (Microsoft Neural Voices)
        edge_voice = _resolve_edge_voice(selected_voice)
        try:
            import edge_tts

            communicate = edge_tts.Communicate(clean_text, edge_voice)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name

            await communicate.save(tmp_path)
            with open(tmp_path, "rb") as f:
                data = f.read()
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if data:
                logger.info(f"Synthesized audio successfully using edge-tts with voice '{edge_voice}'.")
                return data
        except Exception as e:
            logger.warning(f"edge-tts synthesis failed: {e}. Trying secondary gTTS engine...")

        # 2. Secondary Engine: gTTS (Google Text-to-Speech)
        try:
            from gtts import gTTS

            def _run_gtts():
                tts = gTTS(text=clean_text[:4000], lang="en")
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    tmp_path = tmp.name
                tts.save(tmp_path)
                with open(tmp_path, "rb") as f:
                    content = f.read()
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                return content

            data = await asyncio.to_thread(_run_gtts)
            if data:
                logger.info("Synthesized audio successfully using gTTS.")
                return data
        except Exception as e:
            logger.warning(f"gTTS synthesis failed: {e}. Trying macOS native speech engine...")

        # 3. macOS Native Fallback ('say' + 'afconvert')
        try:
            def _run_macos_say():
                aiff_path = tempfile.mktemp(suffix=".aiff")
                wav_path = tempfile.mktemp(suffix=".wav")
                try:
                    subprocess.run(["say", "-o", aiff_path, clean_text[:2000]], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16", aiff_path, wav_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    with open(wav_path, "rb") as f:
                        content = f.read()
                    return content
                finally:
                    if os.path.exists(aiff_path):
                        os.remove(aiff_path)
                    if os.path.exists(wav_path):
                        os.remove(wav_path)

            data = await asyncio.to_thread(_run_macos_say)
            if data:
                logger.info("Synthesized audio successfully using macOS native 'say' engine.")
                return data
        except Exception as e:
            logger.warning(f"macOS native speech synthesis failed: {e}. Trying pyttsx3 engine...")

        # 4. Offline pyttsx3 fallback
        try:
            import pyttsx3

            def _run_pyttsx3():
                engine = pyttsx3.init()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                engine.save_to_file(clean_text[:2000], tmp_path)
                engine.runAndWait()
                with open(tmp_path, "rb") as f:
                    content = f.read()
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                return content

            data = await asyncio.to_thread(_run_pyttsx3)
            if data:
                logger.info("Synthesized audio successfully using pyttsx3 engine.")
                return data
        except Exception as e:
            logger.error(f"pyttsx3 synthesis failed: {e}")

        logger.warning("No operational TTS engine produced audio.")
        return b""

    def _create_azure_service(self) -> Any:
        from app.services.azure_speech_service import AzureSpeechService
        return AzureSpeechService()

    def _create_kokoro_service(self) -> Any:
        from app.services.kokoro_tts_service import KokoroTTSService
        return KokoroTTSService()

    async def synthesize_base64(self, text: str, voice: Optional[str] = None) -> str:
        """Synthesize text into a Base64-encoded audio string."""
        audio_bytes = await self.synthesize_bytes(text, voice)
        if not audio_bytes:
            return ""
        return base64.b64encode(audio_bytes).decode("utf-8")

    async def synthesize_with_metrics(self, text: str, voice: Optional[str] = None) -> Tuple[bytes, float]:
        """Synthesize text and return (audio_bytes, synthesis_time_ms)."""
        import time
        start_time = time.perf_counter()
        audio_bytes = await self.synthesize_bytes(text, voice)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return audio_bytes, round(elapsed_ms, 2)


tts_service = TTSService()
