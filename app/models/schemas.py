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
    provider: Optional[str] = None
    speed: Optional[float] = 1.0
    format: str = "mp3"


class TTSResponse(BaseModel):
    text: str
    audio_base64: str
    format: str = "mp3"


class TTSQualityMetrics(BaseModel):
    # Latency & Performance
    synthesis_time_ms: float = 0.0
    audio_duration_sec: float = 0.0
    real_time_factor: float = 0.0
    time_to_first_chunk_ms: Optional[float] = None

    # Signal & Audio Quality
    sample_rate: int = 24000
    channels: int = 1
    signal_to_noise_ratio_db: float = 0.0
    clipping_ratio: float = 0.0
    peak_amplitude: float = 0.0

    # Faithfulness & Pronunciation
    word_error_rate: float = 0.0
    character_error_rate: float = 0.0
    text_similarity: float = 1.0
    transcribed_text: str = ""

    # Prosody & Rhythm
    mean_f0_hz: Optional[float] = None
    f0_std_dev_hz: Optional[float] = None
    pause_ratio: float = 0.0
    words_per_minute: float = 0.0

    # Overall Evaluation Summary
    overall_quality_pass: bool = True
    quality_score: float = 100.0


class TTSValidationRequest(BaseModel):
    text: str
    expected_text: Optional[str] = None
    stt_text: Optional[str] = None
    voice: Optional[str] = None
    provider: Optional[str] = None
    format: str = "mp3"


class TTSValidationResponse(BaseModel):
    text: str
    expected_text: Optional[str] = None
    tts_ok: bool
    stt_ok: bool
    similarity: float
    word_error_rate: float
    tts_details: dict[str, Any]
    stt_details: dict[str, Any]
    metrics: Optional[TTSQualityMetrics] = None



