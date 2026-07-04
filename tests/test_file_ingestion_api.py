# Creator: Sulabh Bansod
# Description: Integration tests for the file ingestion REST API.
# Use: Validates the POST /ingest/file endpoint by uploading simulated files and verifying indexing.

from fastapi.testclient import TestClient
import pytest
from app.main import app
from app.services.document_service import document_service

client = TestClient(app, raise_server_exceptions=True)



def test_file_ingestion_endpoint(monkeypatch):
    # Mock class-level methods of OllamaEmbeddings because the global service instance is already created
    monkeypatch.setattr(
        "langchain_ollama.OllamaEmbeddings.embed_documents",
        lambda self, texts: [[0.1] * 768 for _ in texts],
    )
    monkeypatch.setattr(
        "langchain_ollama.OllamaEmbeddings.embed_query",
        lambda self, text: [0.1] * 768,
    )


    # Post a mock text file to the endpoint
    response = client.post(
        "/ingest/file",
        files={"file": ("test.txt", b"This is a test document to ingest.", "text/plain")},
    )

    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "success"
    assert res_data["title"] == "test.txt"
    assert "page_id" in res_data

    # Clean up document storage
    page_id = res_data["page_id"]
    if page_id in document_service.documents:
        document_service.delete_document(page_id)
