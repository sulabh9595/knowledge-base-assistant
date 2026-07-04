# Creator: Sulabh Bansod
# Description: Unit tests for the FileLoader class.
# Use: Validates text parsing from TXT, PDF, and DOCX formats under mock inputs.

from unittest.mock import MagicMock
import pytest
from app.loaders.file_loader import FileLoader


def test_file_loader_read_text():
    content = b"Hello, this is a plain text file."
    result = FileLoader.read_text(content)
    assert result == "Hello, this is a plain text file."


def test_file_loader_read_pdf(monkeypatch):
    class DummyPage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class DummyPdfReader:
        def __init__(self, stream):
            self.pages = [DummyPage("Hello"), DummyPage("World")]

    monkeypatch.setattr("app.loaders.file_loader.PdfReader", DummyPdfReader)

    result = FileLoader.read_pdf(b"dummy pdf bytes")
    assert result == "Hello\nWorld"


def test_file_loader_read_docx(monkeypatch):
    class DummyParagraph:
        def __init__(self, text):
            self.text = text

    class DummyDocument:
        def __init__(self, stream):
            self.paragraphs = [
                DummyParagraph("First paragraph"),
                DummyParagraph("Second paragraph"),
            ]

    monkeypatch.setattr("app.loaders.file_loader.DocxDocument", DummyDocument)

    result = FileLoader.read_docx(b"dummy docx bytes")
    assert result == "First paragraph\nSecond paragraph"
