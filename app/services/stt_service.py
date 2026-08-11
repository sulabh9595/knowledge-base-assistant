# Creator: Sulabh Bansod
# Description: Speech-to-Text (STT) service for transcribing audio file bytes into text.
# Use: Provides audio transcription capabilities using faster-whisper.

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from app.config.settings import settings

logger = logging.getLogger(__name__)


class STTService:
    """Service wrapper for Speech-to-Text transcription."""

    def __init__(self, model_size: Optional[str] = None, device: Optional[str] = None, provider: Optional[str] = None, language: Optional[str] = None):
        self.model_size = model_size or getattr(settings, "stt_model_size", "base")
        self.device = device or getattr(settings, "stt_device", "cpu")
        self.provider = provider or getattr(settings, "stt_provider", "faster-whisper")
        self.language = language or getattr(settings, "azure_stt_language", "en-US")
        self._model = None

    @property
    def model(self):
        """Lazy loader for the faster-whisper WhisperModel."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                logger.info(f"Initializing STT model '{self.model_size}' on device '{self.device}'...")
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type="int8" if self.device == "cpu" else "float16"
                )
            except ImportError:
                logger.warning("faster-whisper package is not installed. STT fallback mode active.")
                self._model = False
            except Exception as e:
                logger.error(f"Failed to load WhisperModel: {e}")
                self._model = False
        return self._model

    def transcribe_bytes(self, audio_bytes: bytes, filename: str = "audio.wav") -> Dict[str, Any]:
        """Transcribe raw audio bytes into text with timing segments and detected language."""
        if not audio_bytes:
            return {
                "text": "",
                "language": "en",
                "language_probability": 0.0,
                "duration": 0.0,
                "segments": []
            }

        suffix = os.path.splitext(filename)[1] if filename else ".wav"
        if not suffix:
            suffix = ".wav"

        # Azure Speech provider
        if self.provider == "azure":
            try:
                azure_service = self._create_azure_service()
                azure_result = azure_service.transcribe_bytes(audio_bytes, filename=filename, language=self.language)
                if azure_result.get("text"):
                    return azure_result
            except Exception as exc:
                logger.warning(f"Azure STT failed: {exc}. Falling back to local speech transcription.")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(audio_bytes)
            temp_file.flush()

        try:
            model = self.model
            if model is False or model is None:
                # Fallback response when faster-whisper is not available in environment
                return {
                    "text": "[STT Unavailable: Please install faster-whisper to transcribe audio files.]",
                    "language": "en",
                    "language_probability": 1.0,
                    "duration": 0.0,
                    "segments": []
                }

            segments, info = model.transcribe(temp_path, beam_size=5)

            text_parts: List[str] = []
            segment_data: List[Dict[str, Any]] = []

            for segment in segments:
                cleaned_text = segment.text.strip()
                if cleaned_text:
                    text_parts.append(cleaned_text)
                    segment_data.append({
                        "start": round(segment.start, 2),
                        "end": round(segment.end, 2),
                        "text": cleaned_text
                    })

            full_text = " ".join(text_parts).strip()

            return {
                "text": full_text,
                "language": info.language if hasattr(info, "language") else "en",
                "language_probability": round(getattr(info, "language_probability", 1.0), 3),
                "duration": round(getattr(info, "duration", 0.0), 2),
                "segments": segment_data
            }

        except Exception as e:
            logger.error(f"Error during audio transcription: {e}")
            return {
                "text": "",
                "language": "en",
                "language_probability": 0.0,
                "duration": 0.0,
                "segments": [],
                "error": str(e)
            }
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    def _create_azure_service(self) -> Any:
        from app.services.azure_speech_service import AzureSpeechService
        return AzureSpeechService()


stt_service = STTService()
