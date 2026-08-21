# Creator: Sulabh Bansod
# Description: API router for Text-to-Speech (TTS) synthesis endpoints.
# Use: Handles standalone speech synthesis and audio streaming requests.

import base64
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
import io

from app.models.schemas import TTSRequest, TTSResponse, TTSValidationRequest, TTSValidationResponse
from app.services.audio_validation_service import AudioValidationService
from app.services.tts_service import TTSService, tts_service

router = APIRouter(prefix="/tts", tags=["tts"])


@router.post("/synthesize", response_model=TTSResponse)
async def synthesize_text(request: TTSRequest):
    """Synthesize text into Base64 encoded audio."""
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text payload cannot be empty.")

        service = TTSService(provider=request.provider) if request.provider else tts_service
        audio_base64 = await service.synthesize_base64(request.text, voice=request.voice)
        return TTSResponse(
            text=request.text,
            audio_base64=audio_base64,
            format=request.format
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS synthesis error: {exc}")


@router.post("/stream")
async def stream_audio(request: TTSRequest):
    """Stream synthesized audio binary directly as audio/mpeg or audio/wav."""
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text payload cannot be empty.")

        service = TTSService(provider=request.provider) if request.provider else tts_service
        audio_bytes = await service.synthesize_bytes(request.text, voice=request.voice)
        if not audio_bytes:
            raise HTTPException(status_code=500, detail="Failed to synthesize audio bytes.")

        media_type = "audio/mpeg" if request.format == "mp3" else "audio/wav"
        return StreamingResponse(io.BytesIO(audio_bytes), media_type=media_type)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS streaming error: {exc}")


@router.post("/validate", response_model=TTSValidationResponse)
async def validate_audio_pipeline(request: TTSValidationRequest):
    """Validate the audio pipeline by checking synthesized audio, signal quality, latency, and transcript metrics."""
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text payload cannot be empty.")

        service = TTSService(provider=request.provider) if request.provider else tts_service
        audio_bytes, elapsed_ms = await service.synthesize_with_metrics(request.text, voice=request.voice)

        validator = AudioValidationService()
        expected = request.expected_text or request.text
        stt_transcription = request.stt_text if request.stt_text is not None else expected

        quality_metrics = validator.evaluate_full_tts_quality(
            prompt_text=expected,
            audio_bytes=audio_bytes,
            stt_text=stt_transcription,
            synthesis_time_ms=elapsed_ms,
        )

        tts_result = validator.validate_tts_output(audio_bytes)
        stt_details = validator.validate_stt_output({"text": stt_transcription}, expected)

        return TTSValidationResponse(
            text=request.text,
            expected_text=request.expected_text,
            tts_ok=tts_result["ok"],
            stt_ok=stt_details["ok"],
            similarity=stt_details["similarity"],
            word_error_rate=stt_details["word_error_rate"],
            tts_details=tts_result,
            stt_details=stt_details,
            metrics=quality_metrics,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audio validation error: {exc}")
