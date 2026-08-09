"""Integration test suite for FastAPI TTS API endpoints."""

import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_tts_synthesize_endpoint():
    """Test /tts/synthesize POST endpoint."""
    with patch("app.api.tts.tts_service.synthesize_base64", new_callable=AsyncMock) as mock_synth:
        mock_synth.return_value = "RkFLRV9BVERJT19CQVNFNjQ="

        response = client.post("/tts/synthesize", json={"text": "Hello test", "voice": "en-US-AvaNeural"})

        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Hello test"
        assert data["audio_base64"] == "RkFLRV9BVERJT19CQVNFNjQ="
        assert data["format"] == "mp3"


def test_tts_synthesize_empty_text_error():
    """Test /tts/synthesize error handling with empty text."""
    response = client.post("/tts/synthesize", json={"text": "   "})
    assert response.status_code == 400


def test_rag_query_with_include_audio():
    """Test /rag/query with include_audio=True."""
    with patch("app.api.rag.rag_service.query") as mock_rag, \
         patch("app.api.rag.tts_service.synthesize_base64", new_callable=AsyncMock) as mock_tts:

        mock_rag.return_value = {
            "question": "What is AGY?",
            "answer": "AGY is Google Antigravity platform.",
            "retrieved_documents": []
        }
        mock_tts.return_value = "QUdZX0FVRElPX0JBU0U2NA=="

        response = client.post("/rag/query", json={"question": "What is AGY?", "top_k": 2, "include_audio": True})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "AGY is Google Antigravity platform."
        assert data["audio_base64"] == "QUdZX0FVRElPX0JBU0U2NA=="


def test_langgraph_query_with_include_audio():
    """Test /agent/langgraph/query with include_audio=True."""
    with patch("app.api.langgraph.langgraph_service.ask_question") as mock_agent, \
         patch("app.api.langgraph.tts_service.synthesize_base64", new_callable=AsyncMock) as mock_tts:

        mock_agent.return_value = {
            "question": "Explain graph agent",
            "answer": "Graph agent uses state graphs for multi-step reasoning.",
            "nodes": [],
            "citations": []
        }
        mock_tts.return_value = "R1JBUEhfQVVESU9fQkFTRTY0"

        response = client.post("/agent/langgraph/query", json={"question": "Explain graph agent", "top_k": 2, "include_audio": True})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Graph agent uses state graphs for multi-step reasoning."
        assert data["audio_base64"] == "R1JBUEhfQVVESU9fQkFTRTY0"
