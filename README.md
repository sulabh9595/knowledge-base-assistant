# Knowledge Base Assistant

[![GitHub stars](https://img.shields.io/github/stars/sulabh9595/knowledge-base-assistant?style=flat-square)](https://github.com/sulabh9595/knowledge-base-assistant)
[![GitHub repo size](https://img.shields.io/github/repo-size/sulabh9595/knowledge-base-assistant?style=flat-square)](https://github.com/sulabh9595/knowledge-base-assistant)

A local-first AI knowledge base assistant built with FastAPI, Streamlit, Ollama, Chroma, LangChain, LangGraph, and optional observability tooling.

This repository ingests Confluence pages and local files, stores them locally, and serves both RAG and graph-based reasoning APIs for grounded question answering.

Repository: https://github.com/sulabh9595/knowledge-base-assistant

## Features

- Ingest Confluence spaces into a local knowledge base
- Upload and index local files such as PDF, DOCX, TXT, and Markdown
- Persist documents locally and rebuild indexes on startup
- Query the knowledge base through a standard RAG pipeline
- Use a LangGraph agent for multi-step reasoning and source-grounded answers
- Browse the app through a Streamlit frontend
- Capture tracing and metrics with Langfuse, Prometheus, and Grafana

## Technology stack

- Python 3.9+
- FastAPI backend
- Streamlit frontend
- Ollama LLM (`Qwen3:8b` default)
- Nomic embeddings (`nomic-embed-text`)
- Chroma vector store
- LangChain and LangGraph
- Langfuse tracing (optional)
- Prometheus and Grafana monitoring
- Pytest for automated testing

## Project structure

- `app/` - FastAPI routes, services, models, and RAG pipeline
- `graph/` - LangGraph reasoning and knowledge graph logic
- `frontend/` - Streamlit user interface
- `memory/` - persisted documents and local memory store
- `docker/` - Docker Compose, Prometheus, and Grafana configuration
- `tests/` - unit and integration tests

## Quick start

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
```

2. Create a `.env` file and configure the required values:

```bash
cat > .env <<'EOF'
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=Qwen3:8b
EMBEDDING_MODEL=nomic-embed-text
CONFLUENCE_BASE_URL=https://your-instance.atlassian.net/wiki
CONFLUENCE_EMAIL=your-email@example.com
CONFLUENCE_API_TOKEN=your-token
EOF
```

3. Start the backend:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

4. Start the frontend:

```bash
streamlit run frontend/app.py
```

5. Open the UI at `http://localhost:8501` and the API at `http://127.0.0.1:8000/health`.

## API endpoints

- `GET /health`
- `GET /metrics`
- `POST /ingest/confluence`
- `POST /ingest/file`
- `POST /rag/query`
- `POST /agent/langgraph/query`
- `GET /documents/`
- `GET /documents/{page_id}`
- `PATCH /documents/{page_id}`
- `DELETE /documents/{page_id}`
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

### LangGraph agent query

```bash
curl -X POST http://127.0.0.1:8000/agent/langgraph/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What are the main topics in the knowledge base?","top_k":3}'
```

## Testing

Run the full test suite with:

```bash
pytest
```

## Observability

If enabled in `.env`, the application can emit Langfuse traces and expose Prometheus metrics. Local monitoring assets are available under `docker/`.

## Notes

- The backend reloads persisted documents on startup from `memory/documents.json`.
- The default Ollama model is `Qwen3:8b`.
- The Streamlit UI is a lightweight interface for ingestion and query testing.
