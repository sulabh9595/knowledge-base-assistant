from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional

from app.config.settings import settings


def _build_langfuse_client(**kwargs: Any) -> Any:
    from langfuse import Langfuse

    return Langfuse(**kwargs)


class LangfuseService:
    def __init__(
        self,
        enabled: Optional[bool] = None,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
    ) -> None:
        self.enabled = settings.langfuse_enabled if enabled is None else enabled
        self.public_key = settings.langfuse_public_key if public_key is None else public_key
        self.secret_key = settings.langfuse_secret_key if secret_key is None else secret_key
        self.host = settings.langfuse_host if host is None else host
        self._client: Optional[Any] = None

    def is_enabled(self) -> bool:
        return bool(self.enabled and self.public_key and self.secret_key)

    def _get_client(self) -> Optional[Any]:
        if self._client is None and self.is_enabled():
            self._client = _build_langfuse_client(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host,
            )
        return self._client

    def flush(self) -> None:
        client = self._get_client()
        if client is not None:
            try:
                client.flush()
            except Exception:
                pass

    @contextmanager
    def trace_rag_query(
        self,
        *,
        question: str,
        retrieved_documents: Optional[list[dict[str, Any]]] = None,
        model: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Iterator[Optional[Any]]:
        client = self._get_client()
        if not client:
            yield None
            return

        if hasattr(client, "trace"):
            trace = client.trace(
                name="rag_query",
                input={"question": question},
                metadata=metadata or {},
                model=model,
            )
            try:
                yield trace
            finally:
                trace.update(output={"retrieved_documents": retrieved_documents or []})
                self.flush()
            return

        with client.start_as_current_span(
            name="rag_query",
            input={"question": question},
            metadata=metadata or {},
        ) as trace:
            try:
                yield trace
            finally:
                trace.update(output={"retrieved_documents": retrieved_documents or []})
                self.flush()

    def observe_generation(
        self,
        *,
        name: str,
        prompt: str,
        output: str,
        model: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[Any]:
        client = self._get_client()
        if not client:
            return None

        if hasattr(client, "generation"):
            generation = client.generation(
                name=name,
                model=model,
                input={"prompt": prompt},
                metadata=metadata or {},
            )
            generation.update(output=output)
            self.flush()
            return generation

        with client.start_as_current_observation(
            name=name,
            as_type="generation",
            input={"prompt": prompt},
            metadata=metadata or {},
            model=model,
        ) as generation:
            generation.update(output=output)
            self.flush()
            return generation
