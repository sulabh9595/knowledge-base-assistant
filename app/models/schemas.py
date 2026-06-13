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


class RAGQueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_documents: list[ConfluencePageSummary]


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


class LangGraphQueryResponse(BaseModel):
    question: str
    answer: str
    nodes: list[LangGraphNodeSummary]
    citations: list[LangGraphCitation]
