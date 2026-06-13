from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from app.config.settings import settings
from app.services.metadata_enrichment_service import MetadataEnrichmentService


class DocumentNotFoundError(Exception):
    pass


class DocumentService:
    def __init__(self, store_file: Optional[str] = None, enrich_service: Optional[MetadataEnrichmentService] = None) -> None:
        self.store_file = Path(store_file or settings.memory_store_file)
        self.store_file.parent.mkdir(parents=True, exist_ok=True)
        self.enrich_service = enrich_service or MetadataEnrichmentService()
        self.documents: dict[str, dict[str, Any]] = self._load_documents()

    def _load_documents(self) -> dict[str, dict[str, Any]]:
        if not self.store_file.exists():
            return {}

        try:
            with self.store_file.open("r", encoding="utf-8") as handle:
                documents = json.load(handle)
                return {doc["page_id"]: doc for doc in documents if "page_id" in doc}
        except Exception:
            return {}

    def _persist_documents(self) -> None:
        with self.store_file.open("w", encoding="utf-8") as handle:
            json.dump(list(self.documents.values()), handle, ensure_ascii=False, indent=2)

    def save_documents(self, documents: list[dict[str, Any]]) -> None:
        for document in documents:
            if not document.get("page_id"):
                continue
            enriched = self.enrich_service.enrich(document)
            self.documents[enriched["page_id"]] = enriched
        self._persist_documents()

    def list_documents(self) -> list[dict[str, Any]]:
        return list(self.documents.values())

    def get_document(self, page_id: str) -> dict[str, Any]:
        document = self.documents.get(page_id)
        if document is None:
            raise DocumentNotFoundError(f"Document {page_id} not found")
        return document

    def update_document(self, page_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        document = self.get_document(page_id)
        document = {**document, **updates}
        if "metadata" in updates:
            document["metadata"] = {**document.get("metadata", {}), **updates["metadata"]}
        document = self.enrich_service.enrich(document)
        self.documents[page_id] = document
        self._persist_documents()
        return document

    def delete_document(self, page_id: str) -> None:
        if page_id not in self.documents:
            raise DocumentNotFoundError(f"Document {page_id} not found")
        del self.documents[page_id]
        self._persist_documents()


document_service = DocumentService()
