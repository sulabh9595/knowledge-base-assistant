from __future__ import annotations

from typing import Optional

from app.rag.pipeline import RAGPipeline


class RAGService:
    def __init__(self, pipeline: Optional[RAGPipeline] = None) -> None:
        self.pipeline = pipeline or RAGPipeline()

    def ingest_documents(self, documents: list[dict]) -> None:
        self.pipeline.ingest_documents(documents)

    def query(self, question: str, top_k: int = 3) -> dict:
        return self.pipeline.answer_question(question, top_k=top_k)

    def reindex(self, documents: list[dict]) -> None:
        self.pipeline.reindex_documents(documents)


rag_service = RAGService()
