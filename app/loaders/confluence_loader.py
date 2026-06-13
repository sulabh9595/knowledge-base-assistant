from __future__ import annotations

import html
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

import httpx


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._text_parts: List[str] = []

    def handle_data(self, data: str) -> None:
        self._text_parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._text_parts).strip()


def html_to_text(html_content: str) -> str:
    extractor = _HTMLTextExtractor()
    extractor.feed(html_content)
    text = extractor.get_text()
    return html.unescape(text)


class ConfluenceLoader:
    def __init__(
        self,
        base_url: str,
        email: str,
        api_token: str,
        page_size: int = 50,
        timeout: int = 30,
    ) -> None:
        if not base_url or not email or not api_token:
            raise ValueError("Confluence base URL, email, and API token are required")

        self.base_url = base_url.rstrip("/")
        self.auth = (email, api_token)
        self.page_size = page_size
        self.client = httpx.Client(auth=self.auth, timeout=timeout)

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/wiki/rest/api{path}"
        response = self.client.get(url, params=params or {})
        response.raise_for_status()
        return response.json()

    def fetch_space_pages(self, space_key: str) -> List[Dict[str, Any]]:
        pages: List[Dict[str, Any]] = []
        start = 0

        while True:
            payload = self._get(
                f"/space/{space_key}/content",
                params={
                    "limit": self.page_size,
                    "start": start,
                    "type": "page",
                    "expand": "body.storage,version,metadata.labels",
                },
            )

            page_payload = payload.get("page", payload)
            results = page_payload.get("results", [])
            if not results:
                break

            for page in results:
                pages.append(self._to_document(page))

            if page_payload.get("size", 0) + page_payload.get("start", 0) >= page_payload.get("totalSize", 0):
                break

            start += self.page_size

        return pages

    def _to_document(self, page: Dict[str, Any]) -> Dict[str, Any]:
        storage = page.get("body", {}).get("storage", {})
        content_html = storage.get("value", "")
        return {
            "page_id": page.get("id", ""),
            "title": page.get("title", ""),
            "source_url": page.get("_links", {}).get("webui", ""),
            "text": html_to_text(content_html),
            "metadata": {
                "version": page.get("version", {}).get("number"),
                "space_key": page.get("space", {}).get("key"),
            },
        }
