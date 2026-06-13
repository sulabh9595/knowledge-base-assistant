import httpx

from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    DocumentDetail,
    DocumentListResponse,
    DocumentUpdateRequest,
    ReindexResponse,
)
from app.services.document_service import DocumentNotFoundError, document_service
from app.services.rag_service import rag_service

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("/", response_model=list[DocumentListResponse])
def list_documents():
    return document_service.list_documents()


@router.get("/{page_id}", response_model=DocumentDetail)
def get_document(page_id: str):
    try:
        return document_service.get_document(page_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.patch("/{page_id}", response_model=DocumentDetail)
def update_document(page_id: str, request: DocumentUpdateRequest):
    try:
        updates = request.model_dump(exclude_unset=True)
        return document_service.update_document(page_id, updates)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/{page_id}")
def delete_document(page_id: str):
    try:
        document_service.delete_document(page_id)
        remaining_docs = document_service.list_documents()
        rag_service.reindex(remaining_docs)
        return {"status": "deleted", "page_id": page_id}
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.post("/reindex", response_model=ReindexResponse)
def reindex_documents():
    documents = document_service.list_documents()
    rag_service.reindex(documents)
    return {"status": "reindexed", "document_count": len(documents)}
