from fastapi import APIRouter

from app.api.health import router as health_router
from app.api.ingestion import router as ingestion_router
from app.api.langgraph import router as langgraph_router
from app.api.rag import router as rag_router
from app.api.documents import router as documents_router

router = APIRouter()
router.include_router(health_router)
router.include_router(ingestion_router)
router.include_router(rag_router)
router.include_router(langgraph_router)
router.include_router(documents_router)
