import asyncio

import httpx

from src.config import settings


class HuggingFaceEmbeddingService:
    def __init__(self) -> None:
        self._url = (
            f"https://router.huggingface.co/hf-inference/models/"
            f"{settings.hf_embed_model}/pipeline/feature-extraction"
        )
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {settings.hf_token}"},
            timeout=settings.hf_embed_timeout,
        )

    async def _fetch_embeddings(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.post(
            self._url,
            json={"inputs": texts, "options": {"wait_for_model": True}},
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data and not isinstance(data[0], list):
            data = [data]
        return data

    async def embed(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        if not texts:
            return []
        if not settings.hf_token:
            raise RuntimeError("HF_TOKEN is not configured.")
        size = batch_size or settings.hf_embed_batch_size
        batches = [texts[i : i + size] for i in range(0, len(texts), size)]
        results = await asyncio.gather(*[self._fetch_embeddings(batch) for batch in batches])
        return [vec for batch in results for vec in batch]
