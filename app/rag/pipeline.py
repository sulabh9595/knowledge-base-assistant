from __future__ import annotations

from typing import Dict, List, Optional

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.embeddings.embeddings import EmbeddingProvider
from app.services.llm_service import OllamaService
from app.vectorstore.chroma_store import ChromaStore
from app.vectorstore.interfaces import VectorStoreRepository
from app.config.settings import settings


class RAGPipeline:
    def __init__(
        self,
        llm_service: Optional[OllamaService] = None,
        vector_store: Optional[VectorStoreRepository] = None,
    ) -> None:
        self.embedding_provider = EmbeddingProvider()
        self.vector_store = vector_store or ChromaStore(
            embedding_provider=self.embedding_provider,
            persist_directory=settings.chroma_persist_directory,
            collection_name=settings.chroma_collection_name,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        self.llm_service = llm_service

    def ingest_documents(self, documents: list[dict]) -> None:
        if not documents:
            return

        chunks = self._chunk_documents(documents)
        self.vector_store.add_documents(chunks)

    def answer_question(self, question: str, top_k: int = 3) -> dict:
        retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        relevant_documents = retriever.get_relevant_documents(question)
        retrieved_with_scores = self.vector_store.similarity_search_with_score(question, k=top_k)

        if not relevant_documents:
            retrieved_with_scores = []

        results: List[Dict] = []
        for document, score in retrieved_with_scores:
            metadata = document.metadata or {}
            results.append(
                {
                    "page_id": metadata.get("page_id", ""),
                    "title": metadata.get("title", ""),
                    "source_url": metadata.get("source_url", ""),
                    "text": document.page_content,
                    "metadata": metadata,
                    "similarity_score": float(score),
                }
            )

        context = "\n\n".join(
            f"Title: {doc.get('title')}\nURL: {doc.get('source_url')}\nContent:\n{doc.get('text')}"
            for doc in results
        )
        prompt = self._build_prompt(question, context)
        if self.llm_service is None:
            self.llm_service = OllamaService()
        answer = self.llm_service.generate(prompt)

        return {
            "question": question,
            "answer": answer,
            "retrieved_documents": results,
        }

    def _chunk_documents(self, documents: list[dict]) -> List[Document]:
        chunks: List[Document] = []
        for doc in documents:
            metadata = {
                "page_id": doc.get("page_id", ""),
                "title": doc.get("title", ""),
                "source_url": doc.get("source_url", ""),
                **doc.get("metadata", {}),
            }
            raw_text = doc.get("text", "")
            text_chunks = self.text_splitter.split_text(raw_text)
            for idx, chunk in enumerate(text_chunks):
                chunk_metadata = {**metadata, "chunk_id": f"{metadata.get('page_id', 'unknown')}_{idx}"}
                chunks.append(Document(page_content=chunk, metadata=chunk_metadata))

        return chunks

    def reindex_documents(self, documents: list[dict]) -> None:
        chunks = self._chunk_documents(documents)
        self.vector_store.reindex(chunks)

    def _build_prompt(self, question: str, context: str) -> str:
        if not context:
            context = "No documents were retrieved for the query."

        return (
            "Use the context below to answer the user question accurately. "
            "If the answer is not contained in the context, say that you do not know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )
