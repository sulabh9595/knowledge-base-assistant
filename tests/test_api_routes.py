# Creator: Sulabh Bansod
# Description: Test suite for the REST API routes.
# Use: Validates FastAPI ingestion, RAG, and document management endpoints.

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Knowledge Base Assistant API is running"}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"message": "ok"}


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "kb_rag_query_latency_seconds" in response.text
    assert "kb_llm_generation_latency_seconds" in response.text
    assert "kb_vector_search_latency_seconds" in response.text


