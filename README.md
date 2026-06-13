# Knowledge Base Assistant

[![GitHub stars](https://img.shields.io/github/stars/sulabh9595/knowledge-base-assistant?style=flat-square)](https://github.com/sulabh9595/knowledge-base-assistant)
[![GitHub repo size](https://img.shields.io/github/repo-size/sulabh9595/knowledge-base-assistant?style=flat-square)](https://github.com/sulabh9595/knowledge-base-assistant)

A local-first AI knowledge base assistant built with FastAPI, Streamlit, Ollama, Chroma, and LangGraph.

This repository ingests Confluence pages, stores them locally, and serves both RAG and graph-based reasoning APIs for grounded question answering.

Repository: https://github.com/sulabh9595/knowledge-base-assistant

## Features

- Ingest Confluence spaces into a local knowledge base
- Persist documents to `memory/documents.json`
- Reload persisted data on backend startup
- RAG search via Chroma and Ollama inference
- LangGraph agent reasoning with graph node citations
- Streamlit frontend for ingestion and query testing
- Docker Compose support for local deployment

## Technology stack

- Python 3.12+
- FastAPI backend
- Streamlit frontend
- Ollama LLM (`Qwen3:8b` default)
- Nomic embeddings (`nomic-embed-text`)
- Chroma vector store
- LangGraph-style knowledge graph reasoning
- Pytest for automated testing

## Repository description

**Local AI knowledge base assistant with FastAPI, Streamlit, Ollama, RAG, and LangGraph.**

## Project structure

- `app/` - FastAPI routes, services, models, and RAG pipeline
- `graph/` - LangGraph reasoning and knowledge graph logic
- `frontend/` - Streamlit user interface
- `memory/` - persisted documents and local memory store
- `docker/` - Docker Compose and container setup
- `tests/` - unit and integration tests

## Quick start

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Set values for:

- `OLLAMA_HOST`
- `OLLAMA_MODEL` (default `Qwen3:8b`)
- `EMBEDDING_MODEL` (default `nomic-embed-text`)
- `CONFLUENCE_BASE_URL`
- `CONFLUENCE_EMAIL`
- `CONFLUENCE_API_TOKEN`

3. Start the backend:

```bash
python -m uvicorn app.main:app --reload
```

4. Start the frontend:

```bash
streamlit run frontend/app.py
```

5. Open the UI at `http://localhost:8501`.

## API endpoints

- `GET /health`
- `POST /ingest/confluence`
- `POST /rag/query`
- `POST /agent/langgraph/query`
- `GET /documents`
- `GET /documents/{id}`
- `PATCH /documents/{id}`
- `DELETE /documents/{id}`
- `POST /documents/reindex`

## Example requests

### Confluence ingestion

```bash
curl -X POST http://127.0.0.1:8000/ingest/confluence \
  -H 'Content-Type: application/json' \
  -d '{"space_key":"YOUR_SPACE_KEY"}'
```

### RAG query

```bash
curl -X POST http://127.0.0.1:8000/rag/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the knowledge base about?","top_k":3}'
```
```

### LangGraph agent query

```bash
curl -X POST http://127.0.0.1:8000/agent/langgraph/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What are the main topics in the knowledge base?","top_k":3}'
```
```

## Tests

Run the full test suite with:

```bash
pytest
```

## Docker

Launch the backend and frontend locally:

```bash
docker compose up --build
```

## Notes

- The backend reloads persisted documents on startup from `memory/documents.json`.
- The default Ollama model is `Qwen3:8b`.
- The Streamlit UI is a lightweight test interface for ingestion and queries.
