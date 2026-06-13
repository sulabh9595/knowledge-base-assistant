from fastapi import FastAPI
from app.api.routes import router
from app.services.document_service import document_service
from app.services.langgraph_agent_service import langgraph_service
from app.services.rag_service import rag_service

app = FastAPI(
    title="Knowledge Base Assistant",
    version="0.1.0",
    description="Modular AI knowledge base application API",
)

app.include_router(router)

@app.on_event("startup")
def load_persisted_documents() -> None:
    documents = document_service.list_documents()
    if documents:
        rag_service.ingest_documents(documents)
        langgraph_service.ingest_documents(documents)

@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok"}
