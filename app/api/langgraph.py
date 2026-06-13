import httpx

from fastapi import APIRouter, HTTPException
from app.models.schemas import LangGraphQueryRequest, LangGraphQueryResponse
from app.services.langgraph_agent_service import langgraph_service

router = APIRouter(prefix="/agent/langgraph", tags=["agent"])


@router.post("/query", response_model=LangGraphQueryResponse)
def query_langgraph(request: LangGraphQueryRequest):
    try:
        return langgraph_service.ask_question(request.question, top_k=request.top_k)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
