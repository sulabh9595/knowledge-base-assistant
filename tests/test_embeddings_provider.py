import pytest

from app.embeddings.embeddings import EmbeddingProvider
from app.config.settings import settings


class FakeOllamaEmbeddings:
    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url

    def embed_documents(self, texts):
        return [[len(text), len(text) * 2] for text in texts]

    def embed_query(self, text):
        return [len(text), len(text) * 2]


def test_embedding_provider_uses_ollama(monkeypatch):
    monkeypatch.setattr(
        "app.embeddings.embeddings.OllamaEmbeddings",
        FakeOllamaEmbeddings,
    )
    settings.ollama_host = "http://localhost:11434"
    settings.embedding_model = "nomic-embed-text"

    provider = EmbeddingProvider()
    document_embeddings = provider.embed_documents(["hello", "world"])
    query_embedding = provider.embed_query("hello")

    assert document_embeddings == [[5, 10], [5, 10]]
    assert query_embedding == [5, 10]
    assert provider.client.model == "nomic-embed-text"
    assert provider.client.base_url == "http://localhost:11434"
