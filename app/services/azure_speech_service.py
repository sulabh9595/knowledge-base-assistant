# Creator: Sulabh Bansod
# Description: Azure AI Speech service wrapper.
# Use: Provides Azure Speech-to-Text and Text-to-Speech as optional audio providers.

import logging
import os
import tempfile
from typing import Any, Dict, Optional

from app.config.settings import settings

logger = logging.getLogger(__name__)


class AzureSpeechService:
    def __init__(self) -> None:
        self.enabled = getattr(settings, "azure_speech_enabled", False)
        self.key = getattr(settings, "azure_speech_key", "")
        self.region = getattr(settings, "azure_speech_region", "")
        self.endpoint = getattr(settings, "azure_speech_endpoint", "")
        self.default_tts_voice = getattr(settings, "azure_speech_tts_voice", "en-US-JennyNeural")
        self.default_stt_language = getattr(settings, "azure_stt_language", "en-US")

        if self.enabled and not self.key:
            raise ValueError("Azure Speech is enabled but AZURE_SPEECH_KEY is not configured.")

        if self.enabled and not (self.region or self.endpoint):
            raise ValueError("Azure Speech is enabled but Azure region or endpoint is not configured.")

    def _build_speech_config(self) -> Any:
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError as exc:
            raise ImportError("Azure Cognitive Services Speech SDK is not installed.") from exc

        if self.endpoint:
            speech_config = speechsdk.SpeechConfig(subscription=self.key, region=self.region)
            if hasattr(speech_config, "endpoint"):
                speech_config.endpoint = self.endpoint
            elif hasattr(speech_config, "set_property"):
                speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_Endpoint, self.endpoint)
        else:
            speech_config = speechsdk.SpeechConfig(subscription=self.key, region=self.region)

        return speech_config

    def _get_audio_format(self, audio_format: str) -> Any:
        import azure.cognitiveservices.speech as speechsdk

        format_name = audio_format.lower()
        if format_name == "mp3":
            return speechsdk.SpeechSynthesisOutputFormat.Audio16Khz64KBitRateMonoMp3
        if format_name == "wav":
            return speechsdk.SpeechSynthesisOutputFormat.Audio16Khz16BitMonoPcm

        return speechsdk.SpeechSynthesisOutputFormat.Audio16Khz64KBitRateMonoMp3

    def synthesize_bytes(self, text: str, voice: Optional[str] = None, audio_format: str = "mp3") -> bytes:
        if not self.enabled:
            raise RuntimeError("Azure Speech is not enabled in settings.")

        import azure.cognitiveservices.speech as speechsdk

        voice_name = voice or self.default_tts_voice
        speech_config = self._build_speech_config()
        speech_config.speech_synthesis_voice_name = voice_name
        speech_config.set_speech_synthesis_output_format(self._get_audio_format(audio_format))

        audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=False)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return result.audio_data

        if result.reason == speechsdk.ResultReason.Canceled:
            cancellation = speechsdk.CancellationDetails.from_result(result)
            raise RuntimeError(f"Azure Speech synthesis canceled: {cancellation.reason} - {cancellation.error_details}")

        raise RuntimeError(f"Azure Speech synthesis failed with reason: {result.reason}")

    def transcribe_bytes(self, audio_bytes: bytes, filename: str = "audio.wav", language: Optional[str] = None) -> Dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("Azure Speech is not enabled in settings.")

        import azure.cognitiveservices.speech as speechsdk

        language = language or self.default_stt_language
        suffix = os.path.splitext(filename)[1] or ".wav"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(audio_bytes)
            temp_file.flush()

        try:
            speech_config = self._build_speech_config()
            speech_config.speech_recognition_language = language
            audio_config = speechsdk.audio.AudioConfig(filename=temp_path)
            recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
            result = recognizer.recognize_once_async().get()

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return {
                    "text": result.text,
                    "language": language,
                    "language_probability": 1.0,
                    "duration": 0.0,
                    "segments": []
                }

            if result.reason == speechsdk.ResultReason.Canceled:
                cancellation = speechsdk.CancellationDetails.from_result(result)
                raise RuntimeError(f"Azure STT canceled: {cancellation.reason} - {cancellation.error_details}")

            raise RuntimeError(f"Azure STT failed with reason: {result.reason}")

        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
