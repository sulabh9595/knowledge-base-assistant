# Creator: Sulabh Bansod
# Description: Loader service for reading local uploaded files (PDF, DOCX, TXT).
# Use: Parses file bytes to extract plain text content.

import io
from typing import Any
from pypdf import PdfReader
from docx import Document as DocxDocument


class FileLoader:
    @staticmethod
    def read_text(file_bytes: bytes) -> str:
        """Decode and read standard text files."""
        return file_bytes.decode("utf-8", errors="ignore")

    @staticmethod
    def read_pdf(file_bytes: bytes) -> str:
        """Extract plain text from PDF pages using pypdf."""
        pdf_file = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_file)
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n".join(text_parts).strip()

    @staticmethod
    def read_docx(file_bytes: bytes) -> str:
        """Extract plain text from Word (.docx) documents."""
        docx_file = io.BytesIO(file_bytes)
        doc = DocxDocument(docx_file)
        text_parts = [p.text for p in doc.paragraphs if p.text]
        return "\n".join(text_parts).strip()

    @staticmethod
    def read_audio(file_bytes: bytes, filename: str = "audio.wav") -> dict:
        """Extract plain text transcription and metadata from audio file bytes."""
        from app.services.stt_service import stt_service
        return stt_service.transcribe_bytes(file_bytes, filename=filename)

