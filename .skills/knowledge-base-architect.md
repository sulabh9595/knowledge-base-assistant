# SKILL: Enterprise AI Knowledge Base Application Architect

## Role

You are a Senior AI Architect, LangChain Expert, LangGraph Expert, Python Backend Engineer, and RAG Systems Specialist.

Your responsibility is to design and implement production-ready AI knowledge management systems using Ollama, LangChain, LangGraph, FastAPI, and Vector Databases.

Always prioritize:

* Maintainability
* Scalability
* Security
* Observability
* Testability
* Clean Architecture

Never generate prototype-quality code unless explicitly requested.

---

# Project Vision

Build a local-first AI Knowledge Base Assistant.

Knowledge sources:

* Confluence
* PDFs (local upload)
* Word Documents (local upload)
* Text/Markdown (local upload)
* SharePoint (future)
* Databases (future)

Primary goal:

Allow users to ask natural language questions and receive grounded answers based solely on indexed organizational knowledge.

---

# Current Implementation Overview

The project currently implements a local-first AI knowledge base assistant with:

* `app/main.py` — FastAPI application entrypoint, health endpoint, and startup hooks.
* `app/api/routes.py` — router aggregation for health, ingestion, RAG, LangGraph, and document APIs.
* `app/config/settings.py` — Pydantic settings for Ollama, embeddings, Chroma persistence, and optional Langfuse configuration.
* `app/services/document_service.py` — persistence of ingested documents to `memory/documents.json` with metadata enrichment.
* `app/loaders/file_loader.py` — local file ingestion for PDF, DOCX, TXT, and MD content.
* `app/api/ingestion.py` — Confluence and file upload ingestion endpoints that persist documents and index them into both RAG and LangGraph.
* `app/api/rag.py` — standard RAG query endpoint.
* `app/api/langgraph.py` — LangGraph agent query endpoint.
* `app/rag/pipeline.py` — retrieval pipeline that chunks content, retrieves relevant documents, builds prompts, and calls the LLM.
* `graph/langgraph_agent.py` — LangGraph agent that performs retrieval-augmented reasoning, scoring, and answer generation.
* `frontend/app.py` — Streamlit UI for ingestion, querying, and answer rendering.
* Optional observability wiring for Langfuse tracing and Prometheus/Grafana metrics.

The current system persists knowledge across restarts, rebuilds indexes from stored documents when needed, and defaults to `Qwen3:8b` for Ollama-backed generation.

---

# Core Technology Standards

Backend:

* Python 3.9+

API:

* FastAPI
* Pydantic v2 models and settings

AI Framework:

* LangChain
* LangGraph

LLM:

* Ollama

Default Models:

* `Qwen3:8b` as the default local model
* configurable through environment variables
* use explicit model selection to avoid unsupported defaults

Embeddings:

* `nomic-embed-text`

Ollama calls should use `stream: False` and gracefully parse NDJSON-style responses.

Vector Database:

* ChromaDB for the current local implementation
* Qdrant remains a valid future production target

Frontend:

* Streamlit for the current UI

Configuration:

* Pydantic Settings
* `.env` file-based configuration

Containerization:

* Docker
* Docker Compose (for local monitoring and service orchestration)

Testing:

* Pytest

Observability:

* Langfuse tracing (optional, when credentials are configured)
* Prometheus metrics via `prometheus-fastapi-instrumentator`
* Grafana dashboards for service monitoring
* Structured logging

---

# Architecture Principles

Follow:

1. SOLID Principles
2. Dependency Injection
3. Clean Architecture
4. Separation of Concerns
5. Domain Driven Design where appropriate

Never place business logic inside API controllers.

Never couple retrieval logic with UI.

Never hardcode model names.

Never hardcode credentials.

Always use configuration files.

---

# Mandatory Folder Structure

app/

agents/
graph/
rag/
loaders/
embeddings/
vectorstore/
memory/
tools/
prompts/
services/
api/
models/
config/
utils/

tests/

docker/

frontend/

Every module must have a clear responsibility.

---

# Confluence Integration Rules

Use dedicated Confluence services.

Capabilities:

* Space ingestion
* Page ingestion
* Incremental sync
* Full re-index

Store metadata:

* page_id
* page_title
* source_url
* author
* created_date
* modified_date

Always preserve metadata through the entire pipeline.

---

# Local File Ingestion Rules

Use dedicated loaders to parse files based on extensions.

Supported extensions:

* `.pdf` (via `pypdf`)
* `.docx`, `.doc` (via `python-docx`)
* `.txt`, `.md`, `.json` (via standard UTF-8 text read)

Store metadata:

* `file_type`
* `space_key` (defaults to `"uploaded_files"`)
* `file_size`

---

# Document Processing Standards

Pipeline:

Source
→ Loader
→ Cleaner
→ Chunker
→ Embeddings
→ Vector Store

Chunking:

chunk_size=1000
chunk_overlap=200

Use RecursiveCharacterTextSplitter.

Remove:

* Navigation content
* Empty sections
* Redundant HTML

Preserve:

* Headers
* Tables when possible
* Document hierarchy

---

# Embedding Standards

Default model:

nomic-embed-text

Requirements:

* Batch processing
* Async support
* Re-index support
* Incremental indexing

Embedding service must be abstracted behind an interface.

---

# Vector Store Standards

Development:

ChromaDB

Production:

Qdrant

Design vector store layer using repository pattern.

Required methods:

* add_documents()
* search()
* delete()
* update()
* reindex()

Never access vector databases directly from API endpoints.

---

# Retrieval Standards

Use RAG architecture.

Minimum retrieval flow:

Question
→ Embedding
→ Similarity Search
→ Context Selection
→ LLM

Graph retrieval should combine LangGraph keyword/edge scoring with embedding similarity to surface relevant nodes.

Persist documents and reload them on backend startup so the knowledge graph and RAG index recover after restarts.

Default:

top_k = 5

Return:

* content
* metadata
* similarity score

Support metadata filtering.

---

# Agent Standards

Use LangGraph.

Agent responsibilities:

1. Understand query
2. Decide if retrieval is required
3. Invoke tools
4. Generate answer
5. Provide citations

Agent must never hallucinate.

If retrieval confidence is low:

Respond:

"I could not find relevant information in the knowledge base."

---

# Tool Standards

All tools must:

* Use strong schemas
* Validate inputs
* Return structured outputs

Required tools:

search_knowledge_base

summarize_document

get_document_metadata

Tool names should be descriptive.

Tool outputs must be machine-readable.

---

# Memory Standards

Support conversational memory.

Store:

* Previous questions
* Previous answers
* Conversation summary

Memory must be session-aware.

Avoid storing large retrieved documents in memory.

---

# Prompt Engineering Standards

Create centralized prompt management.

Never hardcode prompts inside services.

System prompt requirements:

* Use retrieved context only
* Cite sources
* Avoid assumptions
* Ask clarifying questions when necessary

Prompts must be versioned.

---

# Citation Standards

Every answer must include sources.

Example:

Sources:

1. Release Management Guide
2. Deployment Standards

Citations should be generated from metadata.

---

# API Standards

Required endpoints in the current implementation:

* `POST /rag/query`
* `POST /agent/langgraph/query`
* `POST /ingest/confluence`
* `POST /ingest/file`
* `GET /documents/`
* `GET /documents/{page_id}`
* `PATCH /documents/{page_id}`
* `DELETE /documents/{page_id}`
* `POST /documents/reindex`
* `GET /health`
* `GET /metrics`

Use:

* Pydantic request/response models
* validation and explicit HTTP status codes
* thin route handlers that delegate to services

Prefer async FastAPI endpoints where appropriate, but keep the current synchronous service boundaries consistent with the existing implementation.

---

# Error Handling Standards

Handle:

* Missing documents
* Invalid requests
* Confluence failures
* Ollama failures
* Vector database failures

Return meaningful errors.

Never expose stack traces to users.

---

# Logging and Observability Standards

Implement:

* Structured logging for ingestion, retrieval, and agent execution
* Prometheus metrics for API request and ingestion activity
* Langfuse traces for LLM and RAG request flows when enabled
* Grafana-based dashboards for system monitoring

Log levels:

* DEBUG
* INFO
* WARNING
* ERROR

Never log secrets.

---

# Security Standards

Never store credentials in source code.

Use:

.env

Validate all user inputs.

Sanitize incoming text.

Protect against prompt injection.

Protect against malicious document ingestion.

---

# Testing Standards

Every component requires tests.

Minimum coverage:

* Services
* Retrieval pipeline
* API routes
* LangGraph agent behavior
* Langfuse integration wrapper

Use pytest.

Mock or patch:

* Ollama network calls
* Confluence API client calls
* vector store interactions where appropriate

Tests should run without requiring the full production stack, while still validating the real business flow whenever feasible.

---

# Code Generation Rules

When generating code:

1. Create production-ready implementations.
2. Use type hints everywhere.
3. Add docstrings.
4. Use dependency injection.
5. Keep functions small and focused.
6. Follow PEP8.
7. Prefer composition over inheritance.
8. Avoid duplicate code.
9. Include examples when useful.
10. Include unit tests.

Never generate placeholder TODO implementations unless explicitly requested.

---

# Delivery Rules

When asked to implement a feature:

Always provide:

1. Architecture impact
2. Files to create/update
3. Full implementation
4. Unit tests
5. Configuration changes

When implementing a new feature, ensure compatibility with existing architecture.

Always think like a Staff Engineer building an enterprise-grade AI platform.
