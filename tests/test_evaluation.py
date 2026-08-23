import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.evaluation.answer_quality import (
    HALLUCINATION_THRESHOLD,
    judge_available,
    summarize_answer_quality,
)
from src.evaluation.retrieval_metrics import aggregate_scores, score_retrieval
from src.evaluation.runner import gold_set_hash, load_qa_file


class TestRetrievalMetrics:
    def test_score_retrieval_gold_hit(self):
        chunks = [{"content": "The Transformer uses scaled dot-product attention."}]
        refs = ["The Transformer uses scaled dot-product attention for all layers."]
        precision, recall, ctx_precision, ctx_recall = score_retrieval(chunks, refs, True)
        assert precision == 1.0
        assert recall == 1.0
        assert ctx_precision == 1.0
        assert ctx_recall == 1.0

    def test_score_retrieval_gold_miss(self):
        chunks = [{"content": "Unrelated content about databases."}]
        refs = ["The Transformer uses scaled dot-product attention."]
        precision, recall, ctx_precision, ctx_recall = score_retrieval(chunks, refs, True)
        assert precision == 0.0
        assert recall == 0.0
        assert ctx_precision == 0.0
        assert ctx_recall == 0.0

    def test_score_retrieval_keyword_mode(self):
        chunks = [{"content": "BLEU score was 34.8 on the test set."}]
        refs = ["BLEU", "34.8"]
        precision, recall, _, _ = score_retrieval(chunks, refs, False)
        assert precision == 1.0
        assert recall == 1.0

    def test_aggregate_scores(self):
        summary = aggregate_scores([1.0, 0.5], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0])
        assert summary["precision"] == 0.75
        assert summary["recall"] == 0.5
        assert summary["f1"] == pytest.approx(0.6)
        assert summary["context_precision"] == 0.5
        assert summary["context_recall"] == 0.5


class TestRunnerHelpers:
    def test_load_qa_file_gold_format(self, tmp_path: Path):
        path = tmp_path / "gold.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "question": "Q1",
                        "ground_truth": "A1",
                        "ground_truth_contexts": ["context one"],
                        "source_pdf": "doc.pdf",
                    }
                ]
            ),
            encoding="utf-8",
        )
        qa, is_gold = load_qa_file(path)
        assert is_gold is True
        assert qa[0]["references"] == ["context one"]

    def test_load_qa_file_keyword_format(self, tmp_path: Path):
        path = tmp_path / "kw.json"
        path.write_text(
            json.dumps(
                {
                    "questions": [
                        {
                            "question": "Q1",
                            "expected_keywords": ["BLEU", "34.8"],
                            "source_pdf": "doc.pdf",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        qa, is_gold = load_qa_file(path)
        assert is_gold is False
        assert qa[0]["references"] == ["BLEU", "34.8"]

    def test_gold_set_hash_stable(self, tmp_path: Path):
        path = tmp_path / "gold.json"
        path.write_text('[{"question": "q"}]', encoding="utf-8")
        assert gold_set_hash(path) == gold_set_hash(path)


class TestGoldSetGeneration:
    @pytest.mark.asyncio
    async def test_generate_gold_set_requires_hf_token(self, monkeypatch, tmp_path: Path):
        from src.evaluation.gold_set import generate_gold_set

        monkeypatch.setattr("src.evaluation.gold_set.settings.hf_token", "")
        db = MagicMock()
        with pytest.raises(RuntimeError, match="HF_TOKEN"):
            await generate_gold_set(5, db, out_path=tmp_path / "gold.json")

    @pytest.mark.asyncio
    async def test_generate_gold_set_appends_to_existing_file(self, monkeypatch, tmp_path: Path):
        from src.evaluation.gold_set import generate_gold_set

        monkeypatch.setattr("src.evaluation.gold_set.settings.hf_token", "tok")

        out_path = tmp_path / "gold.json"
        out_path.write_text(
            json.dumps([{"question": "existing?", "ground_truth": "yes", "ground_truth_contexts": ["x"]}]),
            encoding="utf-8",
        )

        db = MagicMock()
        db.sample_chunks.return_value = [
            {"filename": "doc.pdf", "page": 1, "chunk_index": 0, "content": "The sky is blue."}
        ]

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "src.evaluation.gold_set.HuggingFaceAdapter.generate",
                AsyncMock(return_value='{"question": "What color is the sky?", "answer": "Blue."}'),
            )
            result = await generate_gold_set(5, db, out_path=out_path)

        assert result["generated"] == 1
        assert result["skipped"] == 0
        assert result["total"] == 2

        saved = json.loads(out_path.read_text(encoding="utf-8"))
        assert len(saved) == 2
        assert saved[-1]["question"] == "What color is the sky?"
        assert saved[-1]["ground_truth_contexts"] == ["The sky is blue."]

    @pytest.mark.asyncio
    async def test_generate_gold_set_skips_unparseable_response(self, monkeypatch, tmp_path: Path):
        from src.evaluation.gold_set import generate_gold_set

        monkeypatch.setattr("src.evaluation.gold_set.settings.hf_token", "tok")

        db = MagicMock()
        db.sample_chunks.return_value = [
            {"filename": "doc.pdf", "page": 1, "chunk_index": 0, "content": "Some passage."}
        ]

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "src.evaluation.gold_set.HuggingFaceAdapter.generate",
                AsyncMock(return_value="not json at all"),
            )
            result = await generate_gold_set(5, db, out_path=tmp_path / "gold.json")

        assert result["generated"] == 0
        assert result["skipped"] == 1

    @pytest.mark.asyncio
    async def test_generate_gold_set_no_chunks(self, monkeypatch, tmp_path: Path):
        from src.evaluation.gold_set import generate_gold_set

        monkeypatch.setattr("src.evaluation.gold_set.settings.hf_token", "tok")
        db = MagicMock()
        db.sample_chunks.return_value = []

        result = await generate_gold_set(5, db, out_path=tmp_path / "gold.json")
        assert result == {"generated": 0, "skipped": 0, "total": 0, "path": str(tmp_path / "gold.json")}


class TestAnswerQualitySummary:
    def test_summarize_answer_quality_includes_hallucination_rate(self):
        results = {
            "hybrid": [
                {
                    "faithfulness_score": 1.0,
                    "relevance_score": 0.9,
                    "source_pdf": "a.pdf",
                },
                {
                    "faithfulness_score": 0.2,
                    "relevance_score": 0.8,
                    "source_pdf": "a.pdf",
                },
            ]
        }
        summary = summarize_answer_quality(results)
        assert summary["hybrid"]["faithfulness"] == pytest.approx(0.6)
        assert summary["hybrid"]["hallucination_rate"] == 0.5
        assert summary["hybrid"]["by_source"]["a.pdf"]["sample_count"] == 2

    def test_judge_available_requires_hf_token(self, monkeypatch):
        monkeypatch.setattr("src.evaluation.answer_quality.settings.hf_token", "")
        assert judge_available() is False
        monkeypatch.setattr("src.evaluation.answer_quality.settings.hf_token", "tok")
        assert judge_available() is True

    def test_hallucination_threshold_constant(self):
        assert HALLUCINATION_THRESHOLD == 0.5


@pytest.mark.asyncio
async def test_run_retrieval_eval_smoke(monkeypatch):
    pytest.importorskip("psycopg")
    import os

    if not os.getenv("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set")

    from src.storage.database import DBManager

    try:
        db = DBManager.create()
    except Exception as exc:
        pytest.skip(f"Cannot connect to database: {exc}")

    fixture_path = Path("tests/retriever-evaluation/fixture_qa.json")
    qa, is_gold = load_qa_file(fixture_path)
    vector = [1.0] + [0.0] * 383
    filename = "__pytest_eval_fixture__.pdf"

    try:
        db.add_upload(filename, "https://example.com/fixture.pdf")
        db.save_chunks(
            filename,
            pages=[1, 2],
            indexes=[0, 1],
            texts=[
                "The Transformer uses scaled dot-product attention for all encoder and decoder layers.",
                "We use six identical layers in the baseline encoder and decoder stacks.",
            ],
            vectors=[vector, vector],
        )

        embedder = MagicMock()
        embedder.embed = AsyncMock(return_value=[vector])
        reranker = MagicMock()
        reranker.rerank = AsyncMock(side_effect=lambda query, chunks, top_k: chunks[:top_k])

        from src.evaluation.runner import run_retrieval_eval

        results = await run_retrieval_eval(qa, is_gold, 5, db, embedder, reranker)
        hybrid = results["hybrid"]
        assert hybrid["recall"] >= 0.5
        assert hybrid["precision"] >= 0.5
    finally:
        try:
            db.remove_upload(filename)
        except FileNotFoundError:
            pass


def test_eval_store_roundtrip(monkeypatch):
    pytest.importorskip("psycopg")
    import os

    if not os.getenv("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set")

    from src.storage.database import DBManager

    try:
        db = DBManager.create()
    except Exception as exc:
        pytest.skip(f"Cannot connect to database: {exc}")

    run_id = db.save_eval_run(
        {"question_count": 2, "gold_set_hash": "abc123"},
        {"hybrid": {"precision": 1.0, "recall": 1.0, "f1": 1.0}},
        {"summary": {"hybrid": {"faithfulness": 0.9}}},
    )
    fetched = db.get_eval_run(run_id)
    assert fetched is not None
    assert fetched["config"]["gold_set_hash"] == "abc123"
    assert fetched["retrieval_results"]["hybrid"]["f1"] == 1.0
    assert fetched["id"] == run_id
