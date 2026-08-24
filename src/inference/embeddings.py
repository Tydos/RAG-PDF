"""Hugging Face Inference API client for text embedding generation."""

import asyncio

import httpx

from src.config import settings


class HFError(Exception):
    """Base exception for Hugging Face embedding API failures."""


class MissingHFToken(HFError):
    """Raised when ``HF_TOKEN`` is not configured."""


class HFAuthError(HFError):
    """Raised when the Hugging Face token is rejected (HTTP 401/403)."""


class InvalidEmbeddingDimensions(HFError):
    """Raised when embedding vector length does not match ``settings.embed_dim``."""


class HFTimeoutError(HFError):
    """Raised when an embedding request times out after retries."""


class HuggingFaceEmbeddingService:
    """Generate text embeddings via the Hugging Face Inference API."""

    def __init__(self) -> None:
        """Initialize the HTTP client and embedding endpoint URL.

        Raises:
            MissingHFToken: If ``settings.hf_token`` is empty.
        """
        self._url = (
            f"https://router.huggingface.co/hf-inference/models/"
            f"{settings.hf_embed_model}/pipeline/feature-extraction"
        )
        if not settings.hf_token:
            raise MissingHFToken(
                "Hugging Face token is not provided. Please set the HF_TOKEN environment variable."
            )

        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {settings.hf_token}"},
            timeout=settings.hf_embed_timeout,
        )

    def _validate_embeddings(self, texts: list[str], vectors: list[list[float]]) -> None:
        """Ensure API output matches input count and ``settings.embed_dim``.

        Args:
            texts: Input texts sent in the request.
            vectors: Parsed embedding vectors from the API.

        Raises:
            InvalidEmbeddingDimensions: If count, shape, or vector length is wrong.
        """
        expected = settings.embed_dim

        if len(vectors) != len(texts):
            raise InvalidEmbeddingDimensions(
                f"Expected {len(texts)} embeddings, got {len(vectors)}"
            )

        for index, vector in enumerate(vectors):
            if not isinstance(vector, list) or not vector:
                raise InvalidEmbeddingDimensions(
                    f"Embedding at index {index} is not a non-empty list"
                )
            if len(vector) != expected:
                raise InvalidEmbeddingDimensions(
                    f"Embedding at index {index} has length {len(vector)}, expected {expected}"
                )
            if not all(isinstance(value, (int, float)) for value in vector):
                raise InvalidEmbeddingDimensions(
                    f"Embedding at index {index} contains non-numeric values"
                )

    async def _fetch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Request embeddings for a batch of texts with transient-error retries.

        Args:
            texts: Input strings to embed in a single API call.

        Returns:
            One embedding vector per input text.

        Raises:
            HFAuthError: On HTTP 401 or 403.
            HFTimeoutError: When the request times out after all retry attempts.
            HFError: When retryable HTTP or network errors are exhausted.
            InvalidEmbeddingDimensions: When vector count or length is invalid.
            httpx.HTTPStatusError: On non-retryable HTTP error responses.
        """
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = await self._client.post(
                    self._url,
                    json={"inputs": texts, "options": {"wait_for_model": True}},
                )

                status = response.status_code
                if status in (401, 403):
                    raise HFAuthError("Invalid Credentials")

                if status in (429, 502, 503, 504):
                    if attempt == max_attempts - 1:
                        raise HFError(
                            f"Hugging Face API returned status code {status} "
                            f"after {max_attempts} attempts"
                        )
                    await asyncio.sleep(2**attempt)
                    continue

                response.raise_for_status()
                data = response.json()
                if isinstance(data, list) and data and not isinstance(data[0], list):
                    data = [data]
                self._validate_embeddings(texts, data)
                return data

            except httpx.TimeoutException as e:
                if attempt == max_attempts - 1:
                    raise HFTimeoutError(f"Embedding request timed out with {e}") from e
                await asyncio.sleep(2**attempt)

            except httpx.RequestError as e:
                if attempt == max_attempts - 1:
                    raise HFError(f"Embedding request failed with {e}") from e
                await asyncio.sleep(2**attempt)

        raise HFError("Failed to fetch embeddings after multiple attempts")

    async def embed(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: Input strings to embed.
            batch_size: Texts per API request; defaults to ``settings.hf_embed_batch_size``.

        Returns:
            Embedding vectors in the same order as ``texts``.

        Raises:
            ValueError: If ``batch_size`` is not a positive integer.
            InvalidEmbeddingDimensions: When a batch response has wrong count or length.
            HFError: If a batched API request fails after retries.
        """
        if not texts:
            return []
        size = batch_size or settings.hf_embed_batch_size
        if size < 1:
            raise ValueError("Batch size must be a positive integer.")

        batches = [texts[i : i + size] for i in range(0, len(texts), size)]
        results = await asyncio.gather(*[self._fetch_embeddings(batch) for batch in batches])
        return [vec for batch in results for vec in batch]
