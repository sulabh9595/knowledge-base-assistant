import os
import uuid
import httpx

from fastapi import APIRouter, HTTPException, UploadFile, File
from app.models.schemas import ConfluenceIngestRequest, ConfluenceIngestResponse, FileIngestResponse
from app.loaders.file_loader import FileLoader
from app.services.confluence_service import ConfluenceIngestionService
from app.services.document_service import document_service
from app.services.langgraph_agent_service import langgraph_service
from app.services.rag_service import rag_service
from app.utils.metrics import DOCUMENTS_INGESTED, INGESTION_FAILURES

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/confluence", response_model=ConfluenceIngestResponse)
def ingest_confluence(request: ConfluenceIngestRequest):
    try:
        service = ConfluenceIngestionService()
        pages = service.fetch_space_pages(request.space_key)
        document_service.save_documents(pages)
        rag_service.ingest_documents(pages)
        langgraph_service.ingest_documents(pages)

        # Record metrics
        DOCUMENTS_INGESTED.labels(source="confluence").inc(len(pages))

        return {
            "space_key": request.space_key,
            "page_count": len(pages),
            "pages": [
                {
                    "page_id": page["page_id"],
                    "title": page["title"],
                    "source_url": service._normalize_url(page["source_url"]),
                }
                for page in pages
            ],
        }
    except httpx.HTTPError as exc:
        INGESTION_FAILURES.labels(source="confluence").inc()
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        INGESTION_FAILURES.labels(source="confluence").inc()
        raise HTTPException(status_code=500, detail=str(exc))


AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}


@router.post("/file", response_model=FileIngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    filename = file.filename or "unknown"
    _, ext = os.path.splitext(filename.lower())

    try:
        content_bytes = await file.read()
        duration = 0.0
        language = "en"

        if ext in (".txt", ".md", ".json"):
            text = FileLoader.read_text(content_bytes)
        elif ext == ".pdf":
            text = FileLoader.read_pdf(content_bytes)
        elif ext in (".docx", ".doc"):
            text = FileLoader.read_docx(content_bytes)
        elif ext in AUDIO_EXTENSIONS:
            audio_data = FileLoader.read_audio(content_bytes, filename=filename)
            text = audio_data.get("text", "")
            duration = audio_data.get("duration", 0.0)
            language = audio_data.get("language", "en")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

        if not text.strip():
            raise HTTPException(status_code=400, detail="The uploaded file has no readable text content.")

        page_id = f"file_{uuid.uuid4().hex[:12]}"
        document = {
            "page_id": page_id,
            "title": filename,
            "source_url": f"file://uploaded/{filename}",
            "text": text,
            "metadata": {
                "file_type": ext,
                "space_key": "uploaded_files",
                "file_size": len(content_bytes),
                "media_type": "audio" if ext in AUDIO_EXTENSIONS else "document",
                "duration_seconds": duration,
                "language": language,
            }
        }

        # Save and index document
        document_service.save_documents([document])
        rag_service.ingest_documents([document])
        langgraph_service.ingest_documents([document])

        # Record metrics
        DOCUMENTS_INGESTED.labels(source="file_upload").inc()

        return {
            "status": "success",
            "page_id": page_id,
            "title": filename,
            "word_count": len(text.split()),
        }

    except HTTPException as exc:
        INGESTION_FAILURES.labels(source="file_upload").inc()
        raise
    except Exception as exc:
        INGESTION_FAILURES.labels(source="file_upload").inc()
        raise HTTPException(status_code=500, detail=str(exc))


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

