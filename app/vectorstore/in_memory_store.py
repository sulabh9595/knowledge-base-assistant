from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from numpy.linalg import norm

from app.embeddings.embeddings import EmbeddingProvider
from app.vectorstore.interfaces import VectorStoreRepository


class InMemoryVectorStore(VectorStoreRepository):
    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self.embedding_provider = embedding_provider
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        texts = [doc["text"] for doc in documents]
        if self.embeddings is None:
            self.embeddings = np.array(self.embedding_provider.embed_documents(texts))
        else:
            new_embeddings = np.array(self.embedding_provider.embed_documents(texts))
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        self.documents.extend(documents)

    def similarity_search_with_score(self, query: str, k: int = 5) -> list[tuple[Dict[str, Any], float]]:
        if self.embeddings is None or not self.documents:
            return []

        query_embedding = np.array(self.embedding_provider.embed_query(query))
        query_norm = norm(query_embedding) + 1e-10

        scored_documents: List[tuple[float, Dict[str, Any]]] = []
        for index, embedding in enumerate(self.embeddings):
            document_norm = norm(embedding) + 1e-10
            score = float(np.dot(embedding, query_embedding) / (document_norm * query_norm))
            scored_documents.append((score, self.documents[index]))

        scored_documents.sort(key=lambda item: item[0], reverse=True)
        return [(document, score) for score, document in scored_documents[:k]]

    def as_retriever(self, search_kwargs: Optional[dict[str, Any]] = None):
        raise NotImplementedError("InMemoryVectorStore does not support LangChain retriever")

    def delete(self, ids: list[str]) -> None:
        raise NotImplementedError("Delete is not implemented for InMemoryVectorStore")

    def update(self, documents: List[Dict[str, Any]]) -> None:
        raise NotImplementedError("Update is not implemented for InMemoryVectorStore")

    def reindex(self, documents: List[Dict[str, Any]]) -> None:
        raise NotImplementedError("Reindex is not implemented for InMemoryVectorStore")
