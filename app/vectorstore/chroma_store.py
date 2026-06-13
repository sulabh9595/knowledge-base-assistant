from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from langchain.docstore.document import Document
from langchain.schema import BaseRetriever
from langchain_chroma import Chroma

from app.embeddings.embeddings import EmbeddingProvider
from app.vectorstore.interfaces import VectorStoreRepository


class ChromaStore(VectorStoreRepository):
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        persist_directory: str,
        collection_name: str = "knowledge_base",
    ) -> None:
        self.embedding_provider = embedding_provider
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self.store = Chroma(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            embedding_function=self.embedding_provider.client,
        )

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
        self.store.add_documents(documents)

    def similarity_search_with_score(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        return self.store.similarity_search_with_score(query, k=k)

    def as_retriever(self, search_kwargs: Optional[dict[str, Any]] = None) -> BaseRetriever:
        return self.store.as_retriever(search_kwargs=search_kwargs or {})

    def delete(self, ids: list[str]) -> None:
        if ids:
            self.store.delete(ids=ids)

    def update(self, documents: List[Document]) -> None:
        if not documents:
            return
        self.store.add_documents(documents)

    def reindex(self, documents: List[Document]) -> None:
        if hasattr(self.store, "delete_collection"):
            try:
                self.store.delete_collection(self.collection_name)
            except Exception:
                pass

        self.store = Chroma(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            embedding_function=self.embedding_provider.client,
        )
        self.add_documents(documents)
