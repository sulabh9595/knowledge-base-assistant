# Enterprise Agentic Knowledge Platform

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Kokoro TTS](https://img.shields.io/badge/Kokoro_TTS-82M_Neural-FF6F00?style=flat-square)](https://huggingface.co/hexgrad/Kokoro-82M)
[![DeepEval](https://img.shields.io/badge/DeepEval-2.0+-5F27CD?style=flat-square)](https://confident-ai.com)
[![LangChain](https://img.shields.io/badge/LangChain-Enabled-1C3C3C?style=flat-square)](https://www.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

A local-first, production-grade AI Knowledge Base Assistant built with **FastAPI**, **Streamlit**, **LangGraph**, **ChromaDB**, **Ollama**, **faster-whisper**, **Kokoro-82M Neural TTS**, **edge-tts**, **gTTS**, and **DeepEval**.

This application ingests Confluence spaces, local documents (`PDF`, `DOCX`, `TXT`, `Markdown`), and local audio recordings (`MP3`, `WAV`, `M4A`, `OGG`, `FLAC`, `AAC`), indexes them into a local vector and graph store, and serves both standard RAG and multi-step agentic reasoning APIs (with full text, voice question, and **Text-to-Audio / TTS spoken output** capabilities). It features an integrated **Kokoro Local Neural TTS Engine**, **TTS Quality & Evaluation Framework** (SNR, Clipping, WER/CER, Prosody, Latency RTF), **DeepEval evaluation harness**, **synthetic dataset generator**, and interactive **Streamlit evaluation dashboard**.

---

## 🌟 Key Features

* **📥 Multi-Source Document & Audio Ingestion**:
  * Ingest live Confluence spaces via Atlassian APIs.
  * Ingest local documents (`PDF`, `DOCX`, `TXT`, `Markdown`).
  * Ingest local meeting recordings and voice notes (`MP3`, `WAV`, `M4A`, `OGG`, `FLAC`, `AAC`) using 100% offline Speech-to-Text (`faster-whisper`).
  * Dedicated `/ingest/audio` endpoint with automatic LLM executive summaries and action items (`Qwen3:8b`).
* **🎤 Voice Query & RAG Engine**:
  * Submit text or voice queries to RAG (`POST /rag/query/audio`) and LangGraph Agent (`POST /agent/langgraph/query/audio`).
  * Native microphone recording (`st.audio_input`) and audio query file upload in the Streamlit frontend.
* **🔊 Kokoro Neural Text-to-Audio (TTS) & Multi-Engine Fallback**:
  * Hear spoken answers automatically for all RAG and LangGraph agent responses.
  * **Kokoro-82M Open-Source Local Neural TTS Engine**: 100% local, privacy-first, ultra-fast speech synthesis with custom voice profiles (`af_heart`, `af_bella`, `am_adam`, `am_michael`, `bf_emma`) and speaking speed multiplier ($0.5\times$ to $2.0\times$).
  * Multi-engine fallback chain: `kokoro` ➔ `edge-tts` (Microsoft Neural Voices) ➔ `azure` (Azure Speech) ➔ `gTTS` (Google Speech) ➔ macOS native `say` ➔ `pyttsx3`.
  * Automatic voice resolution mapping when falling back from Kokoro to Edge TTS (`am_michael` $\to$ `en-US-GuyNeural`, `am_adam` $\to$ `en-US-AndrewNeural`, `af_heart` $\to$ `en-US-AvaNeural`).
  * Dedicated standalone REST API endpoints (`POST /tts/synthesize`, `POST /tts/stream`, and `POST /tts/validate`).
  * Integrated HTML5 audio players (`st.audio`) in the Streamlit frontend with provider, voice, and speed controls.
* **📊 TTS Quality & Performance Evaluation Framework**:
  * **Multi-Dimensional Quality Metrics**: Evaluates Signal-to-Noise Ratio (SNR), Clipping Ratio, Word Error Rate (WER), Character Error Rate (CER), Levenshtein similarity, Prosody ($F_0$ pitch dynamics, Words Per Minute WPM, pause ratio), Real-Time Factor ($\text{RTF} = \frac{\text{Synthesis Time}}{\text{Audio Duration}}$), and composite $0-100$ quality scores.
  * **Automated Benchmark Runner**: Run comprehensive benchmark suites across TTS providers via `python evals/benchmark_tts.py` and inspect automatically saved JSON execution reports at `evals/results/tts_benchmark_results.json`.
* **🧠 Dual Retrieval & Reasoning Engines**:
  * **RAG Pipeline**: Vector similarity retrieval with Chroma vector store and grounded response generation.
  * **LangGraph Agent**: Multi-step graph-based reasoning and document relationship mapping.
* **📁 Bulk Audio Ingestion CLI Utility**:
  * Bulk process and index entire local directories of meeting recordings offline using `scripts/ingest_audio_dir.py`.
* **📊 DeepEval Evaluation Framework & Interactive Dashboard**:
  * **Synthetic Dataset Generator**: Automatically create factual, unsupported, audio transcript, and safety test cases from indexed documents using local LLMs.
  * **Multi-Metric Evaluation Harness**: Quantitative scoring for Faithfulness, Answer Relevancy, Contextual Precision/Recall, Hallucination, Toxicity, Bias, Refusal Quality, and Citation Quality.
  * **Interactive Dashboard**: Track metrics trends over time, compare historical runs, and analyze individual test cases via `frontend/eval_dashboard.py` (`scripts/run_eval_dashboard.sh`).
* **📡 Observability & Tracing**: Native Langfuse trace logging, custom Prometheus metrics at `/metrics`, and ready-to-use Grafana monitoring dashboards.
* **🖥️ Interactive UI**: Modern Streamlit web interface with 5 tabs: Service Health, Document & Audio Ingestion, RAG Query, LangGraph Agent, and Text-to-Audio (TTS) Synthesizer.

---

## 🛠️ Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Backend API** | Python 3.9+, FastAPI, Uvicorn, Pydantic |
| **LLM & Embeddings** | Ollama (`Qwen3:8b` default), Nomic (`nomic-embed-text`) |
| **Speech-to-Text (STT)** | `azure` (Azure Speech, optional), `faster-whisper` (CTranslate2 local C++ engine, `int8` CPU / `float16` GPU) |
| **Text-to-Speech (TTS)** | `kokoro` (Local 82M Neural TTS), `edge-tts` (Neural Voices), `azure` (Azure Speech), `gTTS`, macOS `say`, `pyttsx3` |
| **TTS Evaluation** | `jiwer` (WER/CER), `soundfile`, `numpy`, Signal SNR & Clipping Analysis |
| **Supported Media** | Documents (`PDF`, `DOCX`, `TXT`, `MD`), Audio (`MP3`, `WAV`, `M4A`, `OGG`, `FLAC`, `AAC`) |
| **Vector & Graph Store** | ChromaDB, In-Memory Graph Indexing |
| **Orchestration** | LangChain, LangGraph |
| **Evaluation** | DeepEval 2.0+, Pytest |
| **Frontend UI** | Streamlit (`app.py`, `eval_dashboard.py`) |
| **Monitoring** | Langfuse, Prometheus, Grafana |

---

## 📁 Project Structure

```
.
├── app/
│   ├── api/             # FastAPI routes (ingestion, RAG, agent, tts, documents, health)
│   ├── config/          # Application settings & environment configuration
│   ├── embeddings/     # Embeddings provider wrappers
│   ├── loaders/        # Confluence, document, and audio file loaders (PDF, DOCX, TXT, MD, Audio)
│   ├── models/          # Pydantic schemas (RAGQueryResponse, TTSQualityMetrics, TTSResponse, etc.)
│   ├── prompts/         # LLM system & user prompt templates
│   ├── rag/             # Vector store & RAG pipeline implementation
│   ├── services/        # Kokoro TTS, TTSService, STT (faster-whisper), AudioValidation, Ollama, DeepEval services
│   └── utils/           # Prometheus metrics & logger utilities
├── graph/               # LangGraph agent implementation & knowledge graph structure
├── evals/
│   ├── benchmark_tts.py # Automated TTS Quality & Latency Benchmark Script
│   ├── datasets/        # Evaluation datasets (goldens & synthetic test cases with audio transcripts)
│   ├── results/         # Benchmark outputs (tts_benchmark_results.json)
│   └── scripts/         # DeepEval evaluation harness & synthetic case generator
├── eval_results/        # Output directory for latest.json, history, & metrics.csv
├── frontend/
│   ├── app.py           # Main Streamlit web application (5 tabs: Health, Ingestion, RAG, Agent, TTS)
│   └── eval_dashboard.py# Interactive DeepEval results & trend dashboard
├── memory/              # Local document storage & metadata index (`documents.json`)
├── docker/              # Docker Compose, Prometheus, & Grafana configuration
├── scripts/
│   ├── ingest_audio_dir.py   # Bulk local directory audio ingestion CLI tool
│   └── run_eval_dashboard.sh # Launcher script for DeepEval dashboard
├── .skills/
│   ├── kokoro-tts.md     # Kokoro TTS Integration Specification
│   ├── tts-evaluation.md # TTS Evaluation & Quality Benchmarking Specification
│   ├── text-to-audio.md  # Text-to-Audio (TTS) Integration Plan & Architecture Spec
│   └── audio-input-spec.md# 100% Local Audio Ingestion & Voice Processing Spec
└── tests/               # Automated unit & integration test suite (Kokoro TTS, TTS metrics, STT, RAG, Agent)
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

Clone the repository and install dependencies inside the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
pip install kokoro soundfile jiwer gTTS edge-tts
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```env
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=Qwen3:8b
EMBEDDING_MODEL=nomic-embed-text
STT_MODEL_SIZE=base
STT_DEVICE=cpu
ENABLE_TTS=true
TTS_PROVIDER=kokoro
TTS_DEFAULT_VOICE=af_heart
KOKORO_TTS_ENABLED=true
KOKORO_TTS_VOICE=af_heart
KOKORO_TTS_SPEED=1.0
KOKORO_TTS_LANG_CODE=a
# Optional Azure Speech configuration
AZURE_SPEECH_ENABLED=false
AZURE_SPEECH_KEY=
AZURE_SPEECH_REGION=
AZURE_SPEECH_ENDPOINT=
AZURE_SPEECH_TTS_VOICE=en-US-JennyNeural
CONFLUENCE_BASE_URL=https://your-instance.atlassian.net/wiki
CONFLUENCE_EMAIL=your-email@example.com
CONFLUENCE_API_TOKEN=your-confluence-api-token
# Optional Langfuse & Confident AI keys
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
CONFIDENT_API_KEY=
```

### 3. Run Backend API

Start the FastAPI service using the virtual environment python:

```bash
./.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
*(Or activate `.venv` first: `source .venv/bin/activate && uvicorn app.main:app --host 127.0.0.1 --port 8000`)*

* API Docs (Swagger): `http://127.0.0.1:8000/docs`
* Health Check: `http://127.0.0.1:8000/health`

### 4. Run Streamlit Frontend

In a separate terminal, launch the Streamlit UI:

```bash
./.venv/bin/streamlit run frontend/app.py
```

Open `http://localhost:8501` in your browser.

---

## 🎙️ Audio Ingestion & Voice Processing

### 1. Ingesting Meeting Recordings & Audio Files

#### Dedicated Audio API (`POST /ingest/audio`):
Upload meeting recordings to automatically generate an executive summary and index the full transcript:
```bash
curl -X POST "http://127.0.0.1:8000/ingest/audio?generate_summary=true" \
  -F "file=@/path/to/meeting.mp3"
```

#### Bulk Local Directory Ingestion CLI Script:
Scan and ingest an entire folder of local audio files:
```bash
./.venv/bin/python scripts/ingest_audio_dir.py --dir /path/to/audio/folder --api-url http://127.0.0.1:8000
```

### 2. Asking Voice Questions & Hearing Audio Answers

```bash
# Voice RAG Query with Spoken Audio Output
curl -X POST "http://127.0.0.1:8000/rag/query/audio?top_k=3" \
  -F "file=@/path/to/voice_question.wav"

# Voice LangGraph Agent Query with Spoken Audio Output
curl -X POST "http://127.0.0.1:8000/agent/langgraph/query/audio?top_k=3" \
  -F "file=@/path/to/voice_question.wav"

# Standalone Text-to-Speech Endpoint with Kokoro Provider & Custom Speed
curl -X POST "http://127.0.0.1:8000/tts/synthesize" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, welcome to the knowledge platform.", "provider": "kokoro", "voice": "am_michael", "speed": 1.0}'
```

---

## 📊 TTS Evaluation & Benchmarking Framework

### 1. Validating Synthesized Audio (`POST /tts/validate`)

Validate generated audio, signal quality (SNR, clipping), real-time latency (RTF), and transcript similarity (WER/CER):
```bash
curl -X POST "http://127.0.0.1:8000/tts/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The quick brown fox jumps over the lazy dog.",
    "expected_text": "The quick brown fox jumps over the lazy dog.",
    "provider": "kokoro",
    "voice": "am_michael"
  }'
```

### 2. Running Automated TTS Benchmarks

Run automated TTS quality, latency, and RTF benchmarks across providers:
```bash
./.venv/bin/python evals/benchmark_tts.py
```
Benchmark outputs are displayed in stdout and automatically saved to [`evals/results/tts_benchmark_results.json`](file:///Users/sulabh/Documents/Knowledge%20Base%20Application/evals/results/tts_benchmark_results.json).

---

## 🧪 DeepEval Evaluation Framework & Dashboard

### 1. Generating Synthetic Evaluation Datasets

```bash
./.venv/bin/python evals/scripts/generate_synthetic_data.py --num-docs 5 --output evals/datasets/synthetic_cases.json
```

### 2. Running DeepEval Evaluation Harness

```bash
# Evaluate RAG pipeline against Golden cases
./.venv/bin/python evals/scripts/run_deepeval.py --target rag --dataset goldens --top-k 3

# Evaluate LangGraph Agent against Synthetic cases
./.venv/bin/python evals/scripts/run_deepeval.py --target agent --dataset synthetic --top-k 3
```

### 3. Launching Interactive Evaluation Dashboard

```bash
./.venv/bin/streamlit run frontend/eval_dashboard.py
```

---

## 📡 API Endpoint Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health status and Ollama availability |
| `GET` | `/metrics` | Prometheus metrics scrape endpoint |
| `POST` | `/ingest/confluence` | Fetch and index pages from a Confluence space |
| `POST` | `/ingest/file` | Ingest local PDF, DOCX, TXT, MD, MP3, WAV, M4A, OGG, FLAC, AAC files |
| `POST` | `/ingest/audio` | Dedicated local audio ingestion with LLM executive summary generation |
| `POST` | `/rag/query` | Submit text question to standard RAG pipeline (returns text + audio) |
| `POST` | `/rag/query/audio` | Submit voice/audio question to standard RAG pipeline (returns text + audio) |
| `POST` | `/agent/langgraph/query` | Submit text question to LangGraph reasoning agent (returns text + audio) |
| `POST` | `/agent/langgraph/query/audio` | Submit voice/audio question to LangGraph reasoning agent (returns text + audio) |
| `POST` | `/tts/synthesize` | Synthesize text into base64 encoded audio with `provider`, `voice`, and `speed` controls |
| `POST` | `/tts/stream` | Stream synthesized audio directly as binary `audio/mpeg` or `audio/wav` |
| `POST` | `/tts/validate` | Validate audio signal quality (SNR, clipping), WER/CER, latency (RTF), and return `TTSQualityMetrics` |
| `GET` | `/documents/` | List all indexed documents |
| `GET` | `/documents/{page_id}` | Retrieve specific document details |
| `PATCH` | `/documents/{page_id}` | Update document metadata/content |
| `DELETE` | `/documents/{page_id}` | Remove document from knowledge base |
| `POST` | `/documents/reindex` | Force reindexing of stored documents |

---

## 🧪 Unit & Integration Testing

Run the full pytest suite (including Kokoro TTS, TTS evaluation metrics, STT, RAG, and Agent tests):

```bash
./.venv/bin/pytest
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
