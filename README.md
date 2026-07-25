# Enterprise Agentic Knowledge Platform

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![DeepEval](https://img.shields.io/badge/DeepEval-2.0+-5F27CD?style=flat-square)](https://confident-ai.com)
[![LangChain](https://img.shields.io/badge/LangChain-Enabled-1C3C3C?style=flat-square)](https://www.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

A local-first, production-grade AI Knowledge Base Assistant built with **FastAPI**, **Streamlit**, **LangGraph**, **ChromaDB**, **Ollama**, and **DeepEval**.

This application ingests Confluence spaces and local files (PDF, DOCX, TXT, Markdown), indexes them into a local vector and graph store, and serves both standard RAG and multi-step agentic reasoning APIs. It features an integrated **DeepEval evaluation harness** and **synthetic dataset generator** for continuous benchmarking of answer accuracy, context relevancy, hallucination rate, and safety.

---

## 🌟 Key Features

* **📥 Multi-Source Document Ingestion**: Ingest live Confluence spaces via Atlassian APIs or upload local files (`PDF`, `DOCX`, `TXT`, `Markdown`).
* **🧠 Dual Retrieval & Reasoning Engines**:
  * **RAG Pipeline**: Vector similarity retrieval with Chroma vector store and grounded response generation.
  * **LangGraph Agent**: Multi-step graph-based reasoning and document relationship mapping.
* **📊 DeepEval Evaluation Framework**:
  * **Synthetic Dataset Generator**: Automatically create factual, unsupported, and safety test cases from indexed documents using local LLMs.
  * **Multi-Metric Evaluation Harness**: Quantitative scoring for Faithfulness, Answer Relevancy, Contextual Precision/Recall, Hallucination, Toxicity, Bias, Refusal Quality, and Citation Quality.
  * **Automated Reporting**: Exports test run summaries, JSON reports (`latest.json`), and historical time-series metrics (`metrics.csv`).
* **📡 Observability & Tracing**: Native Langfuse trace logging, custom Prometheus metrics at `/metrics`, and ready-to-use Grafana monitoring dashboards.
* **🖥️ Interactive UI**: Modern Streamlit web interface for document uploads, interactive queries, and health checks.

---

## 🛠️ Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Backend API** | Python 3.9+, FastAPI, Uvicorn, Pydantic |
| **LLM & Embeddings** | Ollama (`Qwen3:8b` default), Nomic (`nomic-embed-text`) |
| **Vector & Graph Store** | ChromaDB, In-Memory Graph Indexing |
| **Orchestration** | LangChain, LangGraph |
| **Evaluation** | DeepEval 2.0+, Pytest |
| **Frontend UI** | Streamlit |
| **Monitoring** | Langfuse, Prometheus, Grafana |

---

## 📁 Project Structure

```
.
├── app/
│   ├── api/             # FastAPI routes (ingestion, RAG, agent, documents, health)
│   ├── config/          # Application settings & environment configuration
│   ├── embeddings/     # Embeddings provider wrappers
│   ├── loaders/        # Confluence & document file loaders (PDF, DOCX, TXT, MD)
│   ├── models/          # Pydantic request/response schemas
│   ├── prompts/         # LLM system & user prompt templates
│   ├── rag/             # Vector store & RAG pipeline implementation
│   ├── services/        # Ollama, Langfuse, and DeepEval core services
│   └── utils/           # Prometheus metrics & logger utilities
├── graph/               # LangGraph agent implementation & knowledge graph structure
├── evals/
│   ├── datasets/        # Evaluation datasets (goldens & synthetic test cases)
│   └── scripts/         # DeepEval evaluation harness & synthetic case generator
├── eval_results/        # Output directory for latest.json, history, & metrics.csv
├── frontend/            # Streamlit web application frontend
├── memory/              # Local document storage & metadata index (`documents.json`)
├── docker/              # Docker Compose, Prometheus, & Grafana configuration
└── tests/               # Automated unit & integration tests
```

---

## 🚀 Quick Start Guide

### Prerequisites

* **Python 3.9+** installed
* **Ollama** installed and running locally (`http://127.0.0.1:11434`)
  ```bash
  ollama pull Qwen3:8b
  ollama pull nomic-embed-text
  ```

### 1. Environment Setup

Clone the repository and install the package in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```env
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=Qwen3:8b
EMBEDDING_MODEL=nomic-embed-text
CONFLUENCE_BASE_URL=https://your-instance.atlassian.net/wiki
CONFLUENCE_EMAIL=your-email@example.com
CONFLUENCE_API_TOKEN=your-confluence-api-token
# Optional Langfuse & Confident AI keys
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
CONFIDENT_API_KEY=
```

### 3. Run Backend API

Start the FastAPI service:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

* API Docs (Swagger): `http://127.0.0.1:8000/docs`
* Health Check: `http://127.0.0.1:8000/health`

### 4. Run Streamlit Frontend

In a separate terminal, launch the Streamlit UI:

```bash
streamlit run frontend/app.py
```

Open `http://localhost:8501` in your browser.

---

## 🧪 Evaluation Framework (DeepEval)

This repository incorporates an automated DeepEval pipeline for evaluating and benchmarking RAG and LangGraph agent responses against ground truth datasets or generated synthetic cases.

### 1. Generating Synthetic Evaluation Datasets

Automatically sample your indexed documents in `memory/documents.json` and generate factual, unsupported, and safety test cases:

```bash
python evals/scripts/generate_synthetic_data.py --num-docs 5 --output evals/datasets/synthetic_cases.json
```

### 2. Running Evaluation Harness

Run the evaluation harness against RAG or Agent pipelines:

```bash
# Evaluate RAG pipeline against Golden cases
python evals/scripts/run_deepeval.py --target rag --dataset goldens --top-k 3

# Evaluate LangGraph Agent against Synthetic cases
python evals/scripts/run_deepeval.py --target agent --dataset synthetic --top-k 3

# Evaluate live FastAPI HTTP endpoints
python evals/scripts/run_deepeval.py --target api-rag --dataset goldens --api-url http://127.0.0.1:8000

# Fast heuristic evaluation (no LLM judge overhead)
python evals/scripts/run_deepeval.py --target rag --dataset goldens --use-heuristics
```

#### Supported Command Line Arguments:
* `--target`: Choice of `rag`, `agent`, `api-rag`, `api-agent`.
* `--dataset`: Choice of `goldens` (`evals/datasets/knowledge_base_cases.json`) or `synthetic` (`evals/datasets/synthetic_cases.json`).
* `--top-k`: Number of context documents to retrieve per question (default: `3`).
* `--api-url`: Base URL for API endpoints when using `api-*` targets (default: `http://127.0.0.1:8000`).
* `--use-heuristics`: Forces fast heuristic evaluation without querying an LLM judge.

### 3. Reviewing Results

Evaluation runs automatically generate reports:
* **`eval_results/latest.json`**: Complete detailed evaluation report of the latest run.
* **`eval_results/history/run_<target>_<dataset>_<timestamp>.json`**: Historical archive per run.
* **`eval_results/metrics.csv`**: Time-series log containing metric scores for trend tracking over time.

---

## 📡 API Endpoint Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health status and Ollama availability |
| `GET` | `/metrics` | Prometheus metrics scrape endpoint |
| `POST` | `/ingest/confluence` | Fetch and index pages from a Confluence space |
| `POST` | `/ingest/file` | Ingest local PDF, DOCX, TXT, or MD files |
| `POST` | `/rag/query` | Submit question to standard RAG pipeline |
| `POST` | `/agent/langgraph/query` | Submit question to LangGraph reasoning agent |
| `GET` | `/documents/` | List all indexed documents |
| `GET` | `/documents/{page_id}` | Retrieve specific document details |
| `PATCH` | `/documents/{page_id}` | Update document metadata/content |
| `DELETE` | `/documents/{page_id}` | Remove document from knowledge base |
| `POST` | `/documents/reindex` | Force reindexing of stored documents |

---

## 🐳 Observability & Monitoring

Launch local Prometheus and Grafana instances using Docker Compose:

```bash
docker compose -f docker/docker-compose.yml up -d
```

* **Prometheus**: `http://localhost:9090`
* **Grafana**: `http://localhost:3000` (Default credentials: `admin` / `admin`)

---

## 🧪 Unit & Integration Testing

Run the full pytest suite:

```bash
pytest
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

