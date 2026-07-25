# SKILL: Audio Input & Ingestion Specification (100% Local-Only STT & Voice RAG)

## Role

You are a Senior AI Systems Architect, Speech Recognition Specialist, FastAPI Engineer, and Multimodal RAG Expert.

Your responsibility is to design and implement production-ready **Local Audio Ingestion & Voice Processing Capabilities** (Speech-to-Text document ingestion, batch audio folder indexing, and voice querying) within the **Enterprise Agentic Knowledge Platform**, enforcing a **100% local-first, offline-only architecture** with zero external cloud dependencies.

---

## 1. 100% Local-First Guarantees & Privacy Controls

All components involved in audio processing operate strictly on local compute:

* **Speech-to-Text Engine**: `faster-whisper` (CTranslate2 port of OpenAI Whisper) running on local CPU/GPU (`compute_type="int8"` on CPU, `"float16"` on CUDA GPU).
* **Text Embeddings**: Local Ollama service using `nomic-embed-text`.
* **LLM Reasoning & Summarization**: Local Ollama service using `Qwen3:8b`.
* **Vector & Document Persistence**: Local ChromaDB (`./chroma_store`) and local JSON metadata (`./memory/documents.json`).
* **Zero External Network Calls**: No audio data or transcripts leave the local machine.

---

## 2. Dual Audio Workflows

### 2.1 Audio Document Ingestion Pipeline (`POST /ingest/file` & `POST /ingest/audio`)

```
[ Local Audio File (.mp3, .wav, .m4a, .ogg, .flac, .aac) ]
                       │
                       ▼
         [ app/services/stt_service.py ]
          (Local faster-whisper STT)
                       │
             (Full Transcript Text)
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
[ Executive Summarization ]   [ Metadata Enrichment ]
 (Local Ollama Qwen3:8b)      • Duration, Language, Segments
        │                             │
        └──────────────┬──────────────┘
                       ▼
       [ Local Knowledge Base Stores ]
        • memory/documents.json
        • Chroma Vector Store
        • LangGraph Agent Store
```

### 2.2 Voice Query Pipeline (`POST /rag/query/audio` & `POST /agent/langgraph/query/audio`)

```
[ Microphone Audio / Query Audio File ] ──► [ Local STT Service ] ──► [ Transcribed Query Text ] ──► [ Local RAG / LangGraph Agent ]
```

---

## 3. Mandatory File Structure

```
app/
├── services/
│   └── stt_service.py          # Local Speech-To-Text wrapper service (faster-whisper)
├── loaders/
│   └── file_loader.py          # File loader supporting .mp3, .wav, .m4a, .ogg, .flac, .aac via STT
├── models/
│   └── schemas.py              # AudioQueryResponse, FileIngestResponse, and RAG schemas
├── api/
│   ├── ingestion.py            # POST /ingest/file & POST /ingest/audio endpoints
│   ├── rag.py                  # POST /rag/query/audio endpoint
│   └── langgraph.py            # POST /agent/langgraph/query/audio endpoint
frontend/
├── app.py                      # Streamlit UI with audio upload, st.audio_input, & 4 tabs
└── eval_dashboard.py           # DeepEval evaluation dashboard UI
scripts/
├── ingest_audio_dir.py         # CLI tool for bulk local directory audio ingestion
└── run_eval_dashboard.sh       # Launcher script for DeepEval dashboard
tests/
├── test_stt_service.py         # Unit tests for local STT service
└── test_audio_ingestion_api.py # Integration tests for audio file upload & voice QA
.skills/
└── audio-input-spec.md         # 100% Local Audio Specification
```

---

## 4. Technical Implementation Specifications

### 4.1 Local STT Service (`app/services/stt_service.py`)

```python
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional
from app.config.settings import settings

logger = logging.getLogger(__name__)


class STTService:
    """100% Local Speech-to-Text service using faster-whisper."""

    def __init__(self, model_size: Optional[str] = None, device: Optional[str] = None):
        self.model_size = model_size or getattr(settings, "stt_model_size", "base")
        self.device = device or getattr(settings, "stt_device", "cpu")
        self._model = None

    @property
    def model(self):
        """Lazy loader for faster-whisper WhisperModel."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                logger.info(f"Initializing STT model '{self.model_size}' on device '{self.device}'...")
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type="int8" if self.device == "cpu" else "float16"
                )
            except ImportError:
                logger.warning("faster-whisper package is not installed. STT fallback mode active.")
                self._model = False
            except Exception as e:
                logger.error(f"Failed to load WhisperModel: {e}")
                self._model = False
        return self._model

    def transcribe_bytes(self, audio_bytes: bytes, filename: str = "audio.wav") -> Dict[str, Any]:
        """Transcribe raw audio bytes into text with segment timing and language detection."""
        if not audio_bytes:
            return {
                "text": "",
                "language": "en",
                "language_probability": 0.0,
                "duration": 0.0,
                "segments": []
            }

        suffix = os.path.splitext(filename)[1] if filename else ".wav"
        if not suffix:
            suffix = ".wav"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(audio_bytes)
            temp_file.flush()

        try:
            model = self.model
            if model is False or model is None:
                return {
                    "text": "[STT Unavailable: Please install faster-whisper to transcribe audio files.]",
                    "language": "en",
                    "language_probability": 1.0,
                    "duration": 0.0,
                    "segments": []
                }

            segments, info = model.transcribe(temp_path, beam_size=5)

            text_parts: List[str] = []
            segment_data: List[Dict[str, Any]] = []

            for segment in segments:
                cleaned_text = segment.text.strip()
                if cleaned_text:
                    text_parts.append(cleaned_text)
                    segment_data.append({
                        "start": round(segment.start, 2),
                        "end": round(segment.end, 2),
                        "text": cleaned_text
                    })

            full_text = " ".join(text_parts).strip()

            return {
                "text": full_text,
                "language": info.language if hasattr(info, "language") else "en",
                "language_probability": round(getattr(info, "language_probability", 1.0), 3),
                "duration": round(getattr(info, "duration", 0.0), 2),
                "segments": segment_data
            }
        except Exception as e:
            logger.error(f"Error during audio transcription: {e}")
            return {
                "text": "",
                "language": "en",
                "language_probability": 0.0,
                "duration": 0.0,
                "segments": [],
                "error": str(e)
            }
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass


stt_service = STTService()
```

---

### 4.2 Dedicated Audio Ingestion API (`POST /ingest/audio` in `app/api/ingestion.py`)

```python
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}


@router.post("/audio")
async def ingest_audio(
    file: UploadFile = File(...),
    generate_summary: bool = True
):
    """Dedicated 100% local endpoint for ingesting audio recordings with executive summary generation."""
    filename = file.filename or "recording.wav"
    _, ext = os.path.splitext(filename.lower())

    if ext not in AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported audio file format: {ext}")

    try:
        content_bytes = await file.read()
        audio_data = FileLoader.read_audio(content_bytes, filename=filename)
        text = audio_data.get("text", "").strip()

        if not text:
            raise HTTPException(status_code=400, detail="The uploaded audio file produced no transcription text.")

        summary = ""
        if generate_summary:
            try:
                from app.services.llm_service import OllamaService
                llm = OllamaService()
                prompt = f"Provide a concise executive summary and key action points for this meeting transcript:\n\n{text[:3000]}"
                summary = llm.generate(prompt)
            except Exception:
                summary = ""

        page_id = f"audio_{uuid.uuid4().hex[:12]}"
        full_document_text = f"Summary:\n{summary}\n\nFull Transcript:\n{text}" if summary else text

        document = {
            "page_id": page_id,
            "title": filename,
            "source_url": f"file://uploaded/audio/{filename}",
            "text": full_document_text,
            "metadata": {
                "file_type": ext,
                "space_key": "uploaded_audio",
                "file_size": len(content_bytes),
                "media_type": "audio",
                "duration_seconds": audio_data.get("duration", 0.0),
                "language": audio_data.get("language", "en"),
                "segment_count": len(audio_data.get("segments", [])),
                "has_summary": bool(summary),
            }
        }

        document_service.save_documents([document])
        rag_service.ingest_documents([document])
        langgraph_service.ingest_documents([document])

        DOCUMENTS_INGESTED.labels(source="audio_upload").inc()

        return {
            "status": "success",
            "page_id": page_id,
            "title": filename,
            "duration_seconds": audio_data.get("duration", 0.0),
            "language": audio_data.get("language", "en"),
            "word_count": len(text.split()),
            "summary": summary
        }

    except HTTPException:
        INGESTION_FAILURES.labels(source="audio_upload").inc()
        raise
    except Exception as exc:
        INGESTION_FAILURES.labels(source="audio_upload").inc()
        raise HTTPException(status_code=500, detail=str(exc))
```

---

### 4.3 Voice Query Endpoints (`app/api/rag.py` & `app/api/langgraph.py`)

#### RAG Voice Query (`POST /rag/query/audio`)
```python
@router.post("/query/audio", response_model=AudioQueryResponse)
async def query_rag_audio(
    file: UploadFile = File(...),
    top_k: int = 3
):
    try:
        audio_bytes = await file.read()
        transcription = stt_service.transcribe_bytes(audio_bytes, filename=file.filename or "query.wav")
        question_text = transcription.get("text", "").strip()

        if not question_text:
            raise HTTPException(status_code=400, detail="Voice query audio could not be transcribed.")

        result = rag_service.query(question_text, top_k=top_k)
        return {
            "transcribed_question": question_text,
            "audio_language": transcription.get("language", "en"),
            "answer": result["answer"],
            "retrieved_documents": result["retrieved_documents"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
```

#### LangGraph Agent Voice Query (`POST /agent/langgraph/query/audio`)
```python
@router.post("/query/audio", response_model=AudioQueryResponse)
async def query_langgraph_audio(
    file: UploadFile = File(...),
    top_k: int = 3
):
    try:
        audio_bytes = await file.read()
        transcription = stt_service.transcribe_bytes(audio_bytes, filename=file.filename or "query.wav")
        question_text = transcription.get("text", "").strip()

        if not question_text:
            raise HTTPException(status_code=400, detail="Voice query audio could not be transcribed.")

        result = langgraph_service.ask_question(question_text, top_k=top_k)
        retrieved_docs = [
            {"page_id": node.get("page_id"), "title": node.get("title"), "source_url": node.get("source_url")}
            for node in result.get("nodes", [])
        ]
        return {
            "transcribed_question": question_text,
            "audio_language": transcription.get("language", "en"),
            "answer": result["answer"],
            "retrieved_documents": retrieved_docs,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
```

---

### 4.4 Bulk Local Directory Audio Ingestion Script (`scripts/ingest_audio_dir.py`)

A local CLI utility to ingest an entire directory of local meeting recordings:

```python
#!/usr/bin/env python3
"""Local directory audio ingestion script."""

import argparse
import sys
from pathlib import Path
import httpx

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}


def ingest_directory(dir_path: str, api_url: str = "http://127.0.0.1:8000", generate_summary: bool = True):
    path = Path(dir_path).resolve()
    if not path.is_dir():
        print(f"Error: {dir_path} is not a valid directory.")
        sys.exit(1)

    print("=" * 60)
    print("      LOCAL AUDIO DIRECTORY BULK INGESTION TOOL      ")
    print("=" * 60)

    audio_files = [f for f in path.glob("**/*") if f.suffix.lower() in AUDIO_EXTENSIONS]

    if not audio_files:
        print("No audio files found matching supported formats.")
        return

    for idx, file_path in enumerate(audio_files, 1):
        print(f"[{idx}/{len(audio_files)}] Processing: {file_path.name}...")
        try:
            with file_path.open("rb") as f:
                files = {"file": (file_path.name, f.read(), "audio/wav")}
                res = httpx.post(
                    f"{api_url}/ingest/audio?generate_summary={str(generate_summary).lower()}",
                    files=files,
                    timeout=600
                )
                if res.status_code == 200:
                    data = res.json()
                    print(f"   ✓ Ingested successfully (page_id: {data.get('page_id')})")
                else:
                    print(f"   ✗ Ingestion failed: {res.text}")
        except Exception as e:
            print(f"   ✗ Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk ingest local audio files directory.")
    parser.add_argument("--dir", type=str, required=True, help="Directory path containing audio files.")
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8000", help="FastAPI backend URL.")
    parser.add_argument("--no-summary", action="store_true", help="Disable executive summary generation.")
    args = parser.parse_args()
    ingest_directory(args.dir, args.api_url, generate_summary=not args.no_summary)
```

---

## 5. UI Integration (Streamlit)

In `frontend/app.py`:
* **Tab 2 ("Document & Audio Ingestion")**: Includes Confluence Space ingestion, Local File Upload (supporting PDF, DOCX, TXT, MD, MP3, WAV, M4A, OGG, FLAC, AAC), and dedicated **Audio Recording / Meeting Ingestion** with executive summary toggle (`Qwen3:8b`) and duration/language/word count metrics.
* **Tab 1 ("RAG Query")**: Features a **Voice / Audio Question** toggle with native `st.audio_input` microphone capture widget and audio file upload.
* **Tab 3 ("LangGraph Agent")**: Features a **Voice / Audio Question** toggle with native `st.audio_input` microphone capture widget and audio file upload.
* **Tab 4 ("Evaluation Dashboard")**: Integrated DeepEval benchmark dashboard available via `scripts/run_eval_dashboard.sh`.

---

## 6. Verification & Quality Assurance

* Unit tests in `tests/test_stt_service.py` verify STT service initialization, byte transcription, duration, and fallback modes.
* Integration tests in `tests/test_audio_ingestion_api.py` verify `/ingest/audio`, `/ingest/file` with audio files, and `/rag/query/audio` / `/agent/langgraph/query/audio` voice queries.
* Executed locally via `pytest`.
* Zero external API tokens required for audio ingestion or transcription.
