# Creator: Sulabh Bansod
# Description: API router for standard RAG query endpoint.
# Use: Handles requests to ask questions using the standard search-and-answer pipeline.

import httpx

from fastapi import APIRouter, HTTPException
from app.models.schemas import RAGQueryRequest, RAGQueryResponse
from app.services.rag_service import rag_service

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=RAGQueryResponse)
def query_rag(request: RAGQueryRequest):
    try:
        return rag_service.query(request.question, top_k=request.top_k)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
