from __future__ import annotations

import httpx
import json

from typing import Optional

from app.config.settings import settings


class OllamaService:
    def __init__(self, host: Optional[str] = None, model: Optional[str] = None, timeout: int = 120) -> None:
        self.host = host or settings.ollama_host
        self.model = model or settings.ollama_model
        self.timeout = timeout
        self.client: Optional[httpx.Client] = None

        if self.host and self.model:
            self.client = httpx.Client(timeout=httpx.Timeout(timeout, connect=10.0))

    def generate(self, prompt: str) -> str:
        if not self.host or not self.model or not self.client:
            raise ValueError("Both Ollama host and model must be configured in settings")

        url = f"{self.host.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "num_predict": 512,
            "temperature": 0.2,
            "stream": False,
        }
        response = self.client.post(url, json=payload)
        response.raise_for_status()

        try:
            data = response.json()
        except json.JSONDecodeError:
            lines = [line for line in response.text.splitlines() if line.strip()]
            if not lines:
                raise
            data = json.loads(lines[-1])

        return self._extract_text(data)

    def _extract_text(self, data: dict) -> str:
        if isinstance(data, dict):
            if "output" in data:
                output = data["output"]
                if isinstance(output, list):
                    return "".join(str(item) for item in output)
                return str(output)

            if "response" in data:
                return str(data["response"])

            if "choices" in data:
                choices = data["choices"]
                if isinstance(choices, list):
                    texts: list[str] = []
                    for choice in choices:
                        if isinstance(choice, dict):
                            for field in ("text", "message", "content", "response"):
                                if field in choice:
                                    texts.append(str(choice[field]))
                                    break
                        else:
                            texts.append(str(choice))
                    return "\n".join(texts).strip()

            if "result" in data:
                return str(data["result"])

        return str(data)
