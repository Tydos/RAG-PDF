"""
Unit tests for embedding and generator inference code.
All external I/O (HTTP) is mocked via AsyncMock on the httpx client — no live APIs needed.
"""

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.inference.embeddings import (
    HFAuthError,
    HFError,
    HFTimeoutError,
    HuggingFaceEmbeddingService,
    InvalidEmbeddingDimensions,
    MissingHFToken,
)
from src.inference.reranker import HuggingFaceReranker
from src.inference.rag_generator import LLMResponseGenerator
from src.inference.prompt_builder import PromptBuilder
from src.inference.llm_adapters import ClaudeAdapter, HuggingFaceAdapter, LLMAuthError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CHUNKS = [
    {"filename": "doc.pdf", "page": 1, "content": "The sky is blue."},
    {"filename": "doc.pdf", "page": 2, "content": "Water is wet."},
]

SAMPLE_HISTORY = [
    {"role": "user", "content": "What color is the sky?"},
    {"role": "assistant", "content": "Blue."},
]


# ===========================================================================
# HuggingFaceEmbeddingService
# ===========================================================================


@pytest.fixture
def embedding_svc(monkeypatch):
    """Construct HuggingFaceEmbeddingService with a token present."""
    monkeypatch.setattr("src.inference.embeddings.settings.hf_token", "test-token")
    return HuggingFaceEmbeddingService()


class TestHuggingFaceEmbeddingService:
    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_list(self, embedding_svc):
        assert await embedding_svc.embed([]) == []

    def test_raises_when_token_missing(self, monkeypatch):
        monkeypatch.setattr("src.inference.embeddings.settings.hf_token", "")
        with pytest.raises(MissingHFToken, match="HF_TOKEN"):
            HuggingFaceEmbeddingService()

    @pytest.mark.asyncio
    async def test_raises_on_invalid_batch_size(self, embedding_svc):
        with pytest.raises(ValueError, match="Batch size"):
            await embedding_svc.embed(["hello"], batch_size=-1)

    @pytest.mark.asyncio
    async def test_single_batch_calls_fetch_once(self, embedding_svc):
        fake_vectors = [[0.1] * 384, [0.2] * 384]
        with patch.object(
            embedding_svc, "_fetch_embeddings", new=AsyncMock(return_value=fake_vectors)
        ) as mock_fetch:
            with patch("src.inference.embeddings.settings") as mock_settings:
                mock_settings.hf_embed_batch_size = 32
                result = await embedding_svc.embed(["text a", "text b"])
        mock_fetch.assert_called_once_with(["text a", "text b"])
        assert result == fake_vectors

    @pytest.mark.asyncio
    async def test_large_input_batches_correctly(self, embedding_svc):
        texts = [f"text {i}" for i in range(5)]
        batch_vectors = [[float(i)] * 384 for i in range(5)]

        async def fake_fetch(batch):
            start = int(batch[0].split()[-1])
            return [batch_vectors[start + j] for j in range(len(batch))]

        with patch.object(embedding_svc, "_fetch_embeddings", side_effect=fake_fetch):
            with patch("src.inference.embeddings.settings") as mock_settings:
                mock_settings.hf_embed_batch_size = 2
                result = await embedding_svc.embed(texts, batch_size=2)

        assert len(result) == 5
        assert result[0] == batch_vectors[0]
        assert result[4] == batch_vectors[4]

    @pytest.mark.asyncio
    async def test_fetch_normalises_single_vector_response(self, embedding_svc):
        """HF API sometimes returns a flat list for a single input; must be wrapped."""
        flat_vector = [0.1] * 384
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = flat_vector
        mock_response.raise_for_status = MagicMock()

        with patch.object(embedding_svc._client, "post", new=AsyncMock(return_value=mock_response)):
            result = await embedding_svc._fetch_embeddings(["single text"])

        assert result == [flat_vector]

    @pytest.mark.asyncio
    async def test_fetch_raises_hf_auth_error_on_401(self, embedding_svc):
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch.object(embedding_svc._client, "post", new=AsyncMock(return_value=mock_response)):
            with pytest.raises(HFAuthError, match="Invalid Credentials"):
                await embedding_svc._fetch_embeddings(["text"])

    @pytest.mark.asyncio
    async def test_fetch_raises_hf_error_after_exhausted_429(self, embedding_svc):
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_post = AsyncMock(return_value=mock_response)

        with (
            patch.object(embedding_svc._client, "post", mock_post),
            patch("src.inference.embeddings.asyncio.sleep", new=AsyncMock()),
        ):
            with pytest.raises(HFError, match="status code 429"):
                await embedding_svc._fetch_embeddings(["text"])

        assert mock_post.await_count == 3

    @pytest.mark.asyncio
    async def test_fetch_retries_429_then_succeeds(self, embedding_svc):
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.json.return_value = [[0.1] * 384]
        ok_response.raise_for_status = MagicMock()
        mock_post = AsyncMock(side_effect=[rate_limited, ok_response])

        with (
            patch.object(embedding_svc._client, "post", mock_post),
            patch("src.inference.embeddings.asyncio.sleep", new=AsyncMock()),
        ):
            result = await embedding_svc._fetch_embeddings(["text"])

        assert result == [[0.1] * 384]
        assert mock_post.await_count == 2

    @pytest.mark.asyncio
    async def test_fetch_raises_hf_timeout_after_exhausted_timeouts(self, embedding_svc):
        mock_post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))

        with (
            patch.object(embedding_svc._client, "post", mock_post),
            patch("src.inference.embeddings.asyncio.sleep", new=AsyncMock()),
        ):
            with pytest.raises(HFTimeoutError, match="timed out"):
                await embedding_svc._fetch_embeddings(["text"])

        assert mock_post.await_count == 3

    @pytest.mark.asyncio
    async def test_fetch_raises_http_status_error_on_other_codes(self, embedding_svc):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "400", request=MagicMock(), response=mock_response
        )

        with patch.object(embedding_svc._client, "post", new=AsyncMock(return_value=mock_response)):
            with pytest.raises(httpx.HTTPStatusError):
                await embedding_svc._fetch_embeddings(["text"])

    def test_validate_embeddings_rejects_wrong_length(self, embedding_svc):
        with pytest.raises(InvalidEmbeddingDimensions, match="length 128"):
            embedding_svc._validate_embeddings(["a"], [[0.1] * 128])

    def test_validate_embeddings_rejects_count_mismatch(self, embedding_svc):
        with pytest.raises(InvalidEmbeddingDimensions, match="Expected 2"):
            embedding_svc._validate_embeddings(["a", "b"], [[0.1] * 384])

    def test_validate_embeddings_rejects_non_numeric_values(self, embedding_svc):
        with pytest.raises(InvalidEmbeddingDimensions, match="non-numeric"):
            embedding_svc._validate_embeddings(["a"], [["not-a-number"] + [0.1] * 383])

    @pytest.mark.asyncio
    async def test_fetch_validates_dimensions(self, embedding_svc):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [[0.1] * 128]
        mock_response.raise_for_status = MagicMock()

        with patch.object(embedding_svc._client, "post", new=AsyncMock(return_value=mock_response)):
            with pytest.raises(InvalidEmbeddingDimensions, match="length 128"):
                await embedding_svc._fetch_embeddings(["text"])


# ===========================================================================
# HuggingFaceReranker
# ===========================================================================


class TestHuggingFaceReranker:
    @pytest.mark.asyncio
    async def test_empty_chunks_returns_empty_list(self):
        reranker = HuggingFaceReranker()
        assert await reranker.rerank("query", [], 5) == []

    @pytest.mark.asyncio
    async def test_reorders_by_score(self):
        reranker = HuggingFaceReranker()
        chunks = [
            {"content": "low", "score": 0.1},
            {"content": "high", "score": 0.9},
            {"content": "mid", "score": 0.5},
        ]
        with patch.object(
            reranker, "_score_candidates", new=AsyncMock(return_value=[0.1, 0.9, 0.5])
        ):
            with patch("src.inference.reranker.settings") as mock_settings:
                mock_settings.hf_token = "tok"
                result = await reranker.rerank("query", chunks, 2)
        assert [item["content"] for item in result] == ["high", "mid"]
        assert result[0]["rerank_score"] == 0.9

    @pytest.mark.asyncio
    async def test_fail_open_on_http_error(self):
        reranker = HuggingFaceReranker()
        chunks = [{"content": "a"}, {"content": "b"}]
        with patch.object(
            reranker, "_score_candidates", new=AsyncMock(side_effect=RuntimeError("down"))
        ):
            with patch("src.inference.reranker.settings") as mock_settings:
                mock_settings.hf_token = "tok"
                result = await reranker.rerank("query", chunks, 1)
        assert result == [{"content": "a"}]

    @pytest.mark.asyncio
    async def test_skips_rerank_without_token(self):
        reranker = HuggingFaceReranker()
        chunks = [{"content": "a"}, {"content": "b"}]
        with patch("src.inference.reranker.settings") as mock_settings:
            mock_settings.hf_token = ""
            result = await reranker.rerank("query", chunks, 1)
        assert result == [{"content": "a"}]


# ===========================================================================
# PromptBuilder
# ===========================================================================


class TestPromptBuilder:
    def test_format_chunks_produces_numbered_blocks(self):
        text = PromptBuilder.format_chunks(SAMPLE_CHUNKS)
        assert "[1] [doc.pdf p.1]" in text
        assert "[2] [doc.pdf p.2]" in text
        assert "The sky is blue." in text
        assert "Water is wet." in text

    def test_format_chunks_handles_missing_fields(self):
        chunks = [{"content": "bare content"}]
        text = PromptBuilder.format_chunks(chunks)
        assert "[1] [unknown p.?]" in text
        assert "bare content" in text

    def test_format_chunks_strips_whitespace(self):
        chunks = [{"filename": "f.pdf", "page": 1, "content": "  padded  "}]
        text = PromptBuilder.format_chunks(chunks)
        assert "padded" in text
        assert "  padded  " not in text

    def test_build_messages_structure(self):
        messages = PromptBuilder.build_messages("What color?", SAMPLE_CHUNKS, [])
        roles = [m["role"] for m in messages]
        assert roles[0] == "system"
        assert roles[-1] == "user"

    def test_build_messages_includes_system_prompt(self):
        messages = PromptBuilder.build_messages("q", SAMPLE_CHUNKS, [])
        assert messages[0]["content"] == PromptBuilder.SYSTEM_PROMPT

    def test_build_messages_embeds_context_in_user_turn(self):
        messages = PromptBuilder.build_messages("What color?", SAMPLE_CHUNKS, [])
        user_content = messages[-1]["content"]
        assert "The sky is blue." in user_content
        assert "What color?" in user_content

    def test_build_messages_includes_history(self):
        messages = PromptBuilder.build_messages("follow-up", SAMPLE_CHUNKS, SAMPLE_HISTORY)
        roles = [m["role"] for m in messages]
        assert roles.count("user") == 2
        assert roles.count("assistant") == 1

    def test_build_messages_caps_history_at_6(self):
        long_history = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        messages = PromptBuilder.build_messages("q", SAMPLE_CHUNKS, long_history)
        # system + 6 history + 1 user question = 8
        assert len(messages) == 8

    def test_build_messages_no_history(self):
        messages = PromptBuilder.build_messages("q", SAMPLE_CHUNKS, [])
        # system + user question only
        assert len(messages) == 2


# ===========================================================================
# HuggingFaceAdapter
# ===========================================================================


class TestHuggingFaceAdapter:
    def _make_adapter(self):
        return HuggingFaceAdapter(model="test-model", token="tok")

    def _mock_response(self, content: str) -> MagicMock:
        mock = MagicMock()
        mock.status_code = 200
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"choices": [{"message": {"content": content}}]}
        return mock

    @pytest.mark.asyncio
    async def test_calls_post(self):
        adapter = self._make_adapter()
        with patch.object(
            adapter._client,
            "post",
            new=AsyncMock(return_value=self._mock_response("  answer text  ")),
        ) as mock_post:
            result = await adapter.generate(
                [{"role": "user", "content": "q"}], max_tokens=100, temperature=0.2
            )
        mock_post.assert_called_once()
        assert result == "answer text"

    @pytest.mark.asyncio
    async def test_passes_correct_params(self):
        adapter = self._make_adapter()
        messages = [{"role": "user", "content": "hello"}]
        with patch.object(
            adapter._client, "post", new=AsyncMock(return_value=self._mock_response("ok"))
        ) as mock_post:
            await adapter.generate(messages, max_tokens=200, temperature=0.5)
        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "test-model"
        assert payload["messages"] == messages
        assert payload["max_tokens"] == 200
        assert payload["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_strips_whitespace_from_response(self):
        adapter = self._make_adapter()
        with patch.object(
            adapter._client,
            "post",
            new=AsyncMock(return_value=self._mock_response("\n  trimmed  \n")),
        ):
            result = await adapter.generate([], 100, 0.2)
        assert result == "trimmed"

    @pytest.mark.asyncio
    async def test_raises_auth_error_on_401(self):
        adapter = self._make_adapter()
        mock_response = MagicMock()
        mock_response.status_code = 401
        with patch.object(adapter._client, "post", new=AsyncMock(return_value=mock_response)):
            with pytest.raises(LLMAuthError, match="invalid or unauthorized"):
                await adapter.generate([{"role": "user", "content": "q"}], 100, 0.2)


# ===========================================================================
# ClaudeAdapter
# ===========================================================================


class TestClaudeAdapter:
    def _make_adapter(self):
        return ClaudeAdapter(model="claude-test", token="tok")

    def _mock_response(self, text: str) -> MagicMock:
        mock = MagicMock()
        mock.status_code = 200
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"content": [{"text": text}]}
        return mock

    @pytest.mark.asyncio
    async def test_extracts_system_message(self):
        adapter = self._make_adapter()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        with patch.object(
            adapter._client, "post", new=AsyncMock(return_value=self._mock_response("answer"))
        ) as mock_post:
            await adapter.generate(messages, max_tokens=100, temperature=0.2)
        payload = mock_post.call_args.kwargs["json"]
        assert payload["system"] == "You are helpful."
        assert all(m["role"] != "system" for m in payload["messages"])

    @pytest.mark.asyncio
    async def test_user_messages_passed_through(self):
        adapter = self._make_adapter()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        with patch.object(
            adapter._client, "post", new=AsyncMock(return_value=self._mock_response("ok"))
        ) as mock_post:
            await adapter.generate(messages, max_tokens=100, temperature=0.2)
        assert len(mock_post.call_args.kwargs["json"]["messages"]) == 3

    @pytest.mark.asyncio
    async def test_returns_empty_string_on_empty_content(self):
        adapter = self._make_adapter()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"content": []}
        with patch.object(adapter._client, "post", new=AsyncMock(return_value=mock_response)):
            result = await adapter.generate([{"role": "user", "content": "q"}], 100, 0.2)
        assert result == ""

    @pytest.mark.asyncio
    async def test_raises_auth_error_on_401(self):
        adapter = self._make_adapter()
        mock_response = MagicMock()
        mock_response.status_code = 401
        with patch.object(adapter._client, "post", new=AsyncMock(return_value=mock_response)):
            with pytest.raises(LLMAuthError, match="invalid or unauthorized"):
                await adapter.generate([{"role": "user", "content": "q"}], 100, 0.2)

    @pytest.mark.asyncio
    async def test_strips_whitespace_from_response(self):
        adapter = self._make_adapter()
        with patch.object(
            adapter._client, "post", new=AsyncMock(return_value=self._mock_response("  trimmed  "))
        ):
            result = await adapter.generate([{"role": "user", "content": "q"}], 100, 0.2)
        assert result == "trimmed"


# ===========================================================================
# LLMResponseGenerator
# ===========================================================================


class TestLLMResponseGenerator:
    def _make_generator(self, llm_response="The answer."):
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value=llm_response)
        return LLMResponseGenerator(llm=mock_llm, max_tokens=200, temperature=0.2), mock_llm

    @pytest.mark.asyncio
    async def test_returns_no_context_answer_when_chunks_empty(self):
        gen, _ = self._make_generator()
        result = await gen.generate("question?", chunks=[], history=[])
        assert "couldn't find" in result.lower()

    @pytest.mark.asyncio
    async def test_delegates_to_llm_adapter(self):
        gen, mock_llm = self._make_generator("42")
        result = await gen.generate("What is the answer?", SAMPLE_CHUNKS, [])
        mock_llm.generate.assert_called_once()
        assert result == "42"

    @pytest.mark.asyncio
    async def test_passes_max_tokens_and_temperature(self):
        gen, mock_llm = self._make_generator("ok")
        await gen.generate("q", SAMPLE_CHUNKS, [])
        call_kwargs = mock_llm.generate.call_args.kwargs
        assert call_kwargs["max_tokens"] == 200
        assert call_kwargs["temperature"] == 0.2

    @pytest.mark.asyncio
    async def test_returns_fallback_on_llm_exception(self):
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=RuntimeError("API down"))
        gen = LLMResponseGenerator(llm=mock_llm, max_tokens=200, temperature=0.2)
        result = await gen.generate("q", SAMPLE_CHUNKS, [])
        assert "couldn't find" in result.lower()

    @pytest.mark.asyncio
    async def test_propagates_auth_error(self):
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            side_effect=LLMAuthError("LLM API key is invalid or unauthorized")
        )
        gen = LLMResponseGenerator(llm=mock_llm, max_tokens=200, temperature=0.2)
        with pytest.raises(LLMAuthError, match="invalid or unauthorized"):
            await gen.generate("q", SAMPLE_CHUNKS, [])

    @pytest.mark.asyncio
    async def test_returns_fallback_on_empty_llm_response(self):
        gen, _ = self._make_generator("")
        result = await gen.generate("q", SAMPLE_CHUNKS, [])
        assert "couldn't find" in result.lower()

    @pytest.mark.asyncio
    async def test_passes_history_to_prompt_builder(self):
        gen, mock_llm = self._make_generator("ok")
        await gen.generate("follow-up", SAMPLE_CHUNKS, SAMPLE_HISTORY)
        messages = mock_llm.generate.call_args.kwargs["messages"]
        contents = [m["content"] for m in messages]
        assert any("What color is the sky?" in c for c in contents)

    @pytest.mark.asyncio
    async def test_chunks_appear_in_prompt(self):
        gen, mock_llm = self._make_generator("ok")
        await gen.generate("q", SAMPLE_CHUNKS, [])
        messages = mock_llm.generate.call_args.kwargs["messages"]
        user_content = messages[-1]["content"]
        assert "The sky is blue." in user_content
        assert "Water is wet." in user_content
