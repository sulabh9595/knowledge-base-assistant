from __future__ import annotations

from typing import Optional, Sequence

from langchain_ollama import OllamaEmbeddings
from ollama._types import ResponseError

from app.config.settings import settings


class EmbeddingProvider:
    def __init__(self, host: Optional[str] = None, model: Optional[str] = None) -> None:
        self.host = host or settings.ollama_host
        self.model = model or settings.embedding_model
        self.client = None

        if self.host and self.model:
            self.client = OllamaEmbeddings(model=self.model, base_url=self.host)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not self.client:
            raise ValueError("Ollama host and embedding model must be configured")
        try:
            return self.client.embed_documents(list(texts))
        except ResponseError as exc:
            raise RuntimeError(
                "Ollama embedding failed. Ensure the Ollama runtime is installed and that the llama-server binary is available. "
                "On macOS with Homebrew, install/reinstall Ollama and build the server binary if needed."
            ) from exc

    def embed_query(self, text: str) -> list[float]:
        if not self.client:
            raise ValueError("Ollama host and embedding model must be configured")
        try:
            return self.client.embed_query(text)
        except ResponseError as exc:
            raise RuntimeError(
                "Ollama query embedding failed. Ensure the Ollama runtime is installed and that the llama-server binary is available. "
                "On macOS with Homebrew, install/reinstall Ollama and build the server binary if needed."
            ) from exc
