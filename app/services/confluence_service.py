from __future__ import annotations

from typing import List, Optional

from app.config.settings import settings
from app.loaders.confluence_loader import ConfluenceLoader


class ConfluenceIngestionService:
    def __init__(self, loader: Optional[ConfluenceLoader] = None) -> None:
        self.loader = loader or ConfluenceLoader(
            base_url=settings.confluence_base_url,
            email=settings.confluence_email,
            api_token=settings.confluence_api_token,
        )

    def fetch_space_pages(self, space_key: str) -> list[dict]:
        return self.loader.fetch_space_pages(space_key)

    def ingest_space(self, space_key: str) -> dict:
        pages = self.fetch_space_pages(space_key)
        return {
            "space_key": space_key,
            "page_count": len(pages),
            "pages": [
                {
                    "page_id": page["page_id"],
                    "title": page["title"],
                    "source_url": self._normalize_url(page["source_url"]),
                }
                for page in pages
            ],
        }

    def _normalize_url(self, source_url: str) -> str:
        if source_url.startswith("http"):
            return source_url

        return f"{settings.confluence_base_url}{source_url}" if source_url else ""
