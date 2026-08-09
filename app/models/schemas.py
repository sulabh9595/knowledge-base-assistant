# Creator: Sulabh Bansod
# Description: Data models and validation schemas using Pydantic.
# Use: Validates API request payloads and structures API responses.

from typing import Any

from typing import Any, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    message: str


class ConfluenceIngestRequest(BaseModel):
    space_key: str


class ConfluencePageSummary(BaseModel):
    page_id: str
    title: str
    source_url: str


class ConfluenceIngestResponse(BaseModel):
    space_key: str
    page_count: int
    pages: list[ConfluencePageSummary]


class RAGQueryRequest(BaseModel):
    question: str
    top_k: int = 3
    include_audio: bool = True


class RAGQueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_documents: list[ConfluencePageSummary]
    audio_base64: Optional[str] = None
    audio_url: Optional[str] = None


class DocumentListResponse(BaseModel):
    page_id: str
    title: str
    source_url: str
    metadata: dict[str, Any]


class DocumentDetail(BaseModel):
    page_id: str
    title: str
    source_url: str
    text: str
    metadata: dict[str, Any]


class DocumentUpdateRequest(BaseModel):
    title: Optional[str] = None
    source_url: Optional[str] = None
    text: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class DocumentUpdateResponse(BaseModel):
    page_id: str
    title: str
    source_url: str
    text: str
    metadata: dict[str, Any]


class ReindexResponse(BaseModel):
    status: str
    document_count: int


class LangGraphNodeSummary(BaseModel):
    page_id: str
    title: str
    source_url: str
    edges: list[str]


class LangGraphCitation(BaseModel):
    page_id: str
    title: str
    source_url: str
    snippet: str


class LangGraphQueryRequest(BaseModel):
    question: str
    top_k: int = 3
    include_audio: bool = True



class LangGraphQueryResponse(BaseModel):
    question: str
    answer: str
    nodes: list[LangGraphNodeSummary]
    citations: list[LangGraphCitation]
    audio_base64: Optional[str] = None
    audio_url: Optional[str] = None


class FileIngestResponse(BaseModel):
    status: str
    page_id: str
    title: str
    word_count: int


class AudioQueryResponse(BaseModel):
    transcribed_question: str
    audio_language: str
    answer: str
    retrieved_documents: list[dict[str, Any]]
    audio_base64: Optional[str] = None
    audio_url: Optional[str] = None


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    format: str = "mp3"


class TTSResponse(BaseModel):
    text: str
    audio_base64: str
    format: str = "mp3"



