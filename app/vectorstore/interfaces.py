from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from langchain.docstore.document import Document
from langchain.schema import BaseRetriever


class VectorStoreRepository(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        raise NotImplementedError

    @abstractmethod
    def similarity_search_with_score(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        raise NotImplementedError

    @abstractmethod
    def as_retriever(self, search_kwargs: Optional[dict[str, Any]] = None) -> BaseRetriever:
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, documents: List[Document]) -> None:
        raise NotImplementedError

    @abstractmethod
    def reindex(self, documents: List[Document]) -> None:
        raise NotImplementedError
