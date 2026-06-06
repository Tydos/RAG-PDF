# LLMAdapter defines a common interface for language model adapters, allowing for different implementations (e.g., Claude, Hugging Face) to be used interchangeably in the inference pipeline.
from abc import ABC, abstractmethod

import httpx

from src.config import settings


class LLMAdapter(ABC):
    @abstractmethod
    async def generate(self, messages: list[dict], max_tokens: int, temperature: float) -> str:
        pass


class ClaudeAdapter(LLMAdapter):
    def __init__(self, model: str, token: str) -> None:
        self._model = model
        self._client = httpx.AsyncClient(
            headers={
                "x-api-key": token,
                "anthropic-version": settings.claude_api_version,
                "content-type": "application/json",
            },
            timeout=60,
        )

    async def generate(self, messages: list[dict], max_tokens: int, temperature: float) -> str:
        system = ""
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append(m)

        payload: dict = {
            "model": self._model,
            "messages": filtered,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            payload["system"] = system

        response = await self._client.post(settings.claude_api_url, json=payload)
        response.raise_for_status()
        data = response.json()
        content = data.get("content", [])
        return content[0]["text"].strip() if content else ""


# Business logic for talking with Hugging Face Inference API
class HuggingFaceAdapter(LLMAdapter):
    def __init__(self, model: str, token: str, timeout: int = 60) -> None:
        self._model = model
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )

    async def generate(self, messages: list[dict], max_tokens: int, temperature: float) -> str:
        response = await self._client.post(
            settings.hf_chat_url,
            json={
                "model": self._model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
