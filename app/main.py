# Creator: Sulabh Bansod
# Description: FastAPI application entrypoint.
# Use: Initializes the server and triggers document loading on startup.

from fastapi import FastAPI
from app.api.routes import router
from app.services.document_service import document_service
from app.services.langgraph_agent_service import langgraph_service
from app.services.rag_service import rag_service
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="Knowledge Base Assistant",
    version="0.1.0",
    description="Modular AI knowledge base application API",
)

app.include_router(router)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.on_event("startup")
def load_persisted_documents() -> None:
    documents = document_service.list_documents()
    if documents:
        rag_service.ingest_documents(documents)
        langgraph_service.ingest_documents(documents)

@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok"}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
