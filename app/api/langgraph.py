# Creator: Sulabh Bansod
# Description: API router for graph agent query endpoint.
# Use: Handles requests to ask questions using the knowledge graph agent reasoning.

import httpx

from fastapi import APIRouter, HTTPException, UploadFile, File
from app.models.schemas import LangGraphQueryRequest, LangGraphQueryResponse, AudioQueryResponse
from app.services.langgraph_agent_service import langgraph_service
from app.services.stt_service import stt_service

router = APIRouter(prefix="/agent/langgraph", tags=["agent"])


@router.post("/query", response_model=LangGraphQueryResponse)
def query_langgraph(request: LangGraphQueryRequest):
    try:
        return langgraph_service.ask_question(request.question, top_k=request.top_k)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.post("/query/audio", response_model=AudioQueryResponse)
async def query_langgraph_audio(
    file: UploadFile = File(...),
    top_k: int = 3
):
    try:
        audio_bytes = await file.read()
        transcription = stt_service.transcribe_bytes(audio_bytes, filename=file.filename or "query.wav")
        question_text = transcription.get("text", "").strip()

        if not question_text:
            raise HTTPException(status_code=400, detail="Voice query audio could not be transcribed.")

        result = langgraph_service.ask_question(question_text, top_k=top_k)
        retrieved_docs = [
            {"page_id": node.get("page_id"), "title": node.get("title"), "source_url": node.get("source_url")}
            for node in result.get("nodes", [])
        ]
        return {
            "transcribed_question": question_text,
            "audio_language": transcription.get("language", "en"),
            "answer": result["answer"],
            "retrieved_documents": retrieved_docs,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

