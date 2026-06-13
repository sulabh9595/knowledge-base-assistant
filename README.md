# Knowledge Base Application

A local-first AI knowledge base assistant scaffold built with FastAPI and modular architecture.

## Project structure

- `app/` - application code
- `agents/` - agent orchestration modules
- `graph/` - graph database and knowledge graph logic
- `rag/` - retrieval-augmented generation pipeline
- `loaders/` - source connectors and ingestion loaders
- `embeddings/` - embeddings adapters
- `vectorstore/` - vector store integrations
- `memory/` - session and conversation memory
- `tools/` - utility agents and helpers
- `frontend/` - future user interface code
- `tests/` - unit and integration tests
- `docker/` - container definitions

## Quick start

1. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

2. Create a `.env` file from `.env.example` and set your Confluence and Ollama credentials.

3. Start the API:

```bash
python -m uvicorn app.main:app --reload
```

4. Open `http://127.0.0.1:8000`.

## Streamlit UI

Start the frontend with:

```bash
streamlit run frontend/app.py
```

Then open `http://localhost:8501`.

## Docker

Build and run both backend and frontend with Docker Compose:

```bash
docker compose up --build
```

This exposes:
- `http://127.0.0.1:8000` for the FastAPI backend
- `http://127.0.0.1:8501` for the Streamlit UI

## Tests

Run the test suite with:

```bash
pytest
```

or via the project script:

```bash
python -m pytest
```

## Confluence ingestion

POST `/ingest/confluence`

Request body:

```json
{
  "space_key": "YOUR_SPACE_KEY"
}
```

The endpoint ingests pages from the specified Confluence space and adds them to the local RAG and LangGraph stores.

## RAG query

POST `/rag/query`

Request body:

```json
{
  "question": "What is the main topic of the knowledge base?",
  "top_k": 3
}
```

The endpoint retrieves the most relevant documents and returns an LLM-generated answer grounded in those sources.

## LangGraph agent query

POST `/agent/langgraph/query`

Request body:

```json
{
  "question": "What are the main concepts covered in the ingested knowledge?",
  "top_k": 3
}
```

The endpoint returns a graph-based reasoning answer along with the top graph nodes that were used by the agent and source citations for the evidence used in the response.
