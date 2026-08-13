import logging

import httpx

from src.config import settings


class HuggingFaceReranker:
    def __init__(self) -> None:
        self._url = (
            f"https://router.huggingface.co/hf-inference/models/"
            f"{settings.hf_rerank_model}"
        )
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {settings.hf_token}"},
            timeout=settings.hf_rerank_timeout,
        )

    async def _score_candidates(self, query: str, chunks: list[dict]) -> list[float]:
        pairs = [[query, chunk.get("content", "")] for chunk in chunks]
        response = await self._client.post(
            self._url,
            json={"inputs": pairs},
        )
        response.raise_for_status()
        data = response.json()

        scores: list[float] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    scores.append(float(item.get("score", item.get("label", 0.0))))
                else:
                    scores.append(float(item))
        else:
            raise ValueError(f"Unexpected reranker response: {data!r}")

        if len(scores) != len(chunks):
            raise ValueError(
                f"Reranker returned {len(scores)} scores for {len(chunks)} chunks"
            )
        return scores

    async def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[dict]:
        if not chunks:
            return []
        if not settings.hf_token:
            return chunks[:top_k]

        try:
            scores = await self._score_candidates(query, chunks)
        except Exception:
            logging.exception("Reranking failed; returning original order")
            return chunks[:top_k]

        ranked = []
        for chunk, score in zip(chunks, scores):
            item = dict(chunk)
            item["rerank_score"] = score
            ranked.append(item)
        ranked.sort(key=lambda item: item.get("rerank_score", 0.0), reverse=True)
        return ranked[:top_k]
