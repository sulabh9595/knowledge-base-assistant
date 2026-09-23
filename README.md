# Knowledge Base Application

A Python application for ingesting documents and audio, indexing them with a local vector store, and querying them through both a standard RAG pipeline and a knowledge-graph LangGraph agent.

The current implementation includes:
- FastAPI backend with route-based ingestion and query APIs
- local document ingestion for text, PDF, DOCX, and audio files
- Confluence page ingestion
- RAG retrieval over Chroma-backed embeddings
- graph-based agent reasoning using the implementation in `graph/langgraph_agent.py`
- TTS synthesis and basic audio validation
- Streamlit UI for interacting with the backend

## Architecture

The app is organized around a small set of modules:

- `app/main.py`: FastAPI app initialization and startup logic
- `app/api/`: route definitions for health, ingestion, RAG, agent, documents, and TTS
- `app/services/`: service layers for document storage, Confluence ingestion, RAG, LangGraph, STT, and TTS
- `app/rag/pipeline.py`: chunking, retrieval, and answer generation logic
- `graph/langgraph_agent.py`: graph knowledge model and agent prompt orchestration
- `app/vectorstore/`: vector store implementation and repository wrapper
- `frontend/app.py`: Streamlit UI
- `tests/`: pytest suite for API and service behavior

## Current feature set

### Document ingestion
- local file upload via `POST /ingest/file`
- Confluence space ingestion via `POST /ingest/confluence`
- audio upload and transcription via `POST /ingest/audio`
- document storage and lookup via `GET /documents`, `GET /documents/{page_id}`

### Retrieval and answer generation
- standard RAG flow through `app/services/rag_service.py` and `app/rag/pipeline.py`
- graph-based answering through `graph/langgraph_agent.py`
- text queries: `POST /rag/query` and `POST /agent/langgraph/query`
- audio queries: `POST /rag/query/audio` and `POST /agent/langgraph/query/audio`

### Speech-to-text and text-to-speech
- STT using `faster-whisper` and optional Azure STT
- TTS via `tts_service.py` with provider fallback logic
- endpoints:
  - `POST /tts/synthesize`
  - `POST /tts/stream`
  - `POST /tts/validate`

### Application entry points
- backend: `uvicorn app.main:app --host 127.0.0.1 --port 8000`
- frontend: `streamlit run frontend/app.py`

## Project structure

```text
.
├── app/
│   ├── api/
│   │   ├── documents.py
│   │   ├── health.py
│   │   ├── ingestion.py
│   │   ├── langgraph.py
│   │   ├── rag.py
│   │   ├── routes.py
│   │   └── tts.py
│   ├── config/
│   │   └── settings.py
│   ├── embeddings/
│   │   └── embeddings.py
│   ├── loaders/
│   │   ├── confluence_loader.py
│   │   └── file_loader.py
│   ├── models/
│   │   └── schemas.py
│   ├── rag/
│   │   └── pipeline.py
│   ├── services/
│   │   ├── audio_validation_service.py
│   │   ├── azure_speech_service.py
│   │   ├── confluence_service.py
│   │   ├── document_service.py
│   │   ├── langfuse_service.py
│   │   ├── langgraph_agent_service.py
│   │   ├── llm_service.py
│   │   ├── rag_service.py
│   │   ├── stt_service.py
│   │   └── tts_service.py
│   └── utils/
├── graph/
│   └── langgraph_agent.py
├── frontend/
│   ├── app.py
│   └── eval_dashboard.py
├── tests/
├── data/
├── memory/
├── chroma_store/
├── docker/
├── evals/
├── eval_results/
├── scripts/
├── app.py
├── requirements.txt
├── pyproject.toml
├── pytest.ini
├── README.md
└── .env.example (if present in your local setup)
```

## Dependencies and environment

The repo currently uses the packages defined in `requirements.txt` and `pyproject.toml`, including:
- FastAPI
- Streamlit
- LangChain / LangGraph
- ChromaDB
- Ollama-based LLM integration
- faster-whisper
- Azure Speech SDK
- edge-tts, gTTS, pyttsx3
- Prometheus instrumentation
- pytest

Configuration is loaded from environment variables through `app/config/settings.py`. The defaults include:
- `OLLAMA_HOST`
- `OLLAMA_MODEL`
- `EMBEDDING_MODEL`
- `CHROMA_PERSIST_DIRECTORY`
- `MEMORY_STORE_FILE`
- `STT_MODEL_SIZE`
- `STT_DEVICE`
- `TTS_PROVIDER`
- `TTS_DEFAULT_VOICE`
- `KOKORO_TTS_ENABLED`
- Confluence credentials if using Confluence ingestion

## Local setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

or:

```bash
pip install -e .
```

### 3. Start Ollama

The app expects Ollama to be available for the LLM and embeddings layer. For example:

```bash
ollama serve
```

You may also need to pull a model used by the app, such as:

```bash
ollama pull Qwen3:8b
ollama pull nomic-embed-text
```

## Run the app

### Backend

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Frontend

```bash
streamlit run frontend/app.py
```

## API examples

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### Ingest a local file

```bash
curl -X POST "http://127.0.0.1:8000/ingest/file" \
  -F "file=@/path/to/sample.pdf"
```

### Ingest audio

```bash
curl -X POST "http://127.0.0.1:8000/ingest/audio?generate_summary=true" \
  -F "file=@/path/to/meeting.wav"
```

### Ask a RAG question

```bash
curl -X POST "http://127.0.0.1:8000/rag/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is this project about?","top_k":3}'
```

### Ask a LangGraph question

```bash
curl -X POST "http://127.0.0.1:8000/agent/langgraph/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize the most relevant documents.","top_k":3}'
```

### Synthesize speech

```bash
curl -X POST "http://127.0.0.1:8000/tts/synthesize" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello from the knowledge base app.","provider":"kokoro","voice":"af_heart"}'
```

## Current endpoints

- `GET /`
- `GET /health`
- `GET /metrics`
- `POST /ingest/confluence`
- `POST /ingest/file`
- `POST /ingest/audio`
- `POST /rag/query`
- `POST /rag/query/audio`
- `POST /agent/langgraph/query`
- `POST /agent/langgraph/query/audio`
- `POST /tts/synthesize`
- `POST /tts/stream`
- `POST /tts/validate`
- `GET /documents/`
- `GET /documents/{page_id}`
- `PATCH /documents/{page_id}`
- `DELETE /documents/{page_id}`
- `POST /documents/reindex`

## Testing

Run the repo test suite with:

```bash
pytest
```

The project includes tests for API routes, ingestion, TTS, STT, embeddings, and RAG behaviors under the `tests/` directory.

## Notes

- This repo is currently built around a local-first workflow and stores indexed data in the local Chroma database and memory files.
- The startup hook in `app/main.py` reloads stored documents from disk into both the RAG and graph pipeline.
- If a service such as TTS or STT is unavailable, the implementation falls back to alternate providers where possible.

## License

This project is intended to be used under the license defined in the repository. Check the repository root for the project license file.
