from __future__ import annotations

import re
from typing import Any, Dict, List
from urllib.parse import urlparse


DEFAULT_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "been",
    "what", "when", "where", "which", "their", "there", "about", "will",
    "your", "were", "also", "such", "then", "than", "into", "over",
}


class MetadataEnrichmentService:
    def enrich(self, document: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(document)
        metadata = dict(document.get("metadata", {}) or {})
        text = document.get("text", "") or ""

        metadata.setdefault("word_count", self._word_count(text))
        metadata.setdefault("summary", self._extract_summary(text))
        metadata.setdefault("source_domain", self._extract_source_domain(document.get("source_url", "")))
        metadata.setdefault("tags", self._extract_tags(text))

        enriched["metadata"] = metadata
        return enriched

    def _word_count(self, text: str) -> int:
        return len([token for token in re.findall(r"\w+", text) if token])

    def _extract_summary(self, text: str, max_length: int = 200) -> str:
        cleaned = " ".join(text.split())
        if not cleaned:
            return ""
        sentence_end = cleaned.find(". ")
        if sentence_end == -1 or sentence_end > max_length:
            return cleaned[:max_length].rstrip() + ("..." if len(cleaned) > max_length else "")
        return cleaned[:sentence_end + 1]

    def _extract_source_domain(self, source_url: str) -> str:
        try:
            parsed = urlparse(source_url)
            return parsed.netloc or ""
        except Exception:
            return ""

    def _extract_tags(self, text: str, max_tags: int = 5) -> List[str]:
        tokens = [token.lower() for token in re.findall(r"\w+", text)]
        counts: dict[str, int] = {}
        for token in tokens:
            if len(token) < 4 or token in DEFAULT_STOPWORDS:
                continue
            counts[token] = counts.get(token, 0) + 1

        sorted_tokens = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [token for token, _ in sorted_tokens[:max_tags]]
