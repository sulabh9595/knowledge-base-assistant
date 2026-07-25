# Creator: Sulabh Bansod
# Description: Integration tests for audio file ingestion and audio query API endpoints.

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@patch("app.loaders.file_loader.FileLoader.read_audio")
@patch("app.services.document_service.document_service.save_documents")
@patch("app.services.rag_service.rag_service.ingest_documents")
@patch("app.services.langgraph_agent_service.langgraph_service.ingest_documents")
def test_ingest_audio_file(mock_langgraph_ingest, mock_rag_ingest, mock_save_docs, mock_read_audio):
    mock_read_audio.return_value = {
        "text": "This is a transcribed meeting recording.",
        "language": "en",
        "duration": 12.4,
        "segments": []
    }

    file_content = b"fake audio binary data"
    response = client.post(
        "/ingest/file",
        files={"file": ("meeting_notes.mp3", file_content, "audio/mpeg")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["title"] == "meeting_notes.mp3"
    assert data["word_count"] == 6


@patch("app.services.stt_service.stt_service.transcribe_bytes")
@patch("app.services.rag_service.rag_service.query")
def test_rag_audio_query_endpoint(mock_rag_query, mock_transcribe):
    mock_transcribe.return_value = {
        "text": "What is the project architecture?",
        "language": "en",
        "duration": 2.1,
        "segments": []
    }
    mock_rag_query.return_value = {
        "question": "What is the project architecture?",
        "answer": "The project uses FastAPI and ChromaDB.",
        "retrieved_documents": [{"page_id": "p1", "title": "Doc 1", "source_url": "url1"}]
    }

    response = client.post(
        "/rag/query/audio?top_k=3",
        files={"file": ("query.wav", b"fake audio data", "audio/wav")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["transcribed_question"] == "What is the project architecture?"
    assert data["answer"] == "The project uses FastAPI and ChromaDB."
    assert len(data["retrieved_documents"]) == 1
