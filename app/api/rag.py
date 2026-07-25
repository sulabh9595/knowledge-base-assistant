# Creator: Sulabh Bansod
# Description: API router for standard RAG query endpoint.
# Use: Handles requests to ask questions using the standard search-and-answer pipeline.

import httpx

from fastapi import APIRouter, HTTPException, UploadFile, File
from app.models.schemas import RAGQueryRequest, RAGQueryResponse, AudioQueryResponse
from app.services.rag_service import rag_service
from app.services.stt_service import stt_service

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=RAGQueryResponse)
def query_rag(request: RAGQueryRequest):
    try:
        return rag_service.query(request.question, top_k=request.top_k)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.post("/query/audio", response_model=AudioQueryResponse)
async def query_rag_audio(
    file: UploadFile = File(...),
    top_k: int = 3
):
    try:
        audio_bytes = await file.read()
        transcription = stt_service.transcribe_bytes(audio_bytes, filename=file.filename or "query.wav")
        question_text = transcription.get("text", "").strip()

        if not question_text:
            raise HTTPException(status_code=400, detail="Voice query audio could not be transcribed.")

        result = rag_service.query(question_text, top_k=top_k)
        return {
            "transcribed_question": question_text,
            "audio_language": transcription.get("language", "en"),
            "answer": result["answer"],
            "retrieved_documents": result["retrieved_documents"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

