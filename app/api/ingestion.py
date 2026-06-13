import httpx

from fastapi import APIRouter, HTTPException
from app.models.schemas import ConfluenceIngestRequest, ConfluenceIngestResponse
from app.services.confluence_service import ConfluenceIngestionService
from app.services.document_service import document_service
from app.services.langgraph_agent_service import langgraph_service
from app.services.rag_service import rag_service

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/confluence", response_model=ConfluenceIngestResponse)
def ingest_confluence(request: ConfluenceIngestRequest):
    try:
        service = ConfluenceIngestionService()
        pages = service.fetch_space_pages(request.space_key)
        document_service.save_documents(pages)
        rag_service.ingest_documents(pages)
        langgraph_service.ingest_documents(pages)

        return {
            "space_key": request.space_key,
            "page_count": len(pages),
            "pages": [
                {
                    "page_id": page["page_id"],
                    "title": page["title"],
                    "source_url": service._normalize_url(page["source_url"]),
                }
                for page in pages
            ],
        }
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
