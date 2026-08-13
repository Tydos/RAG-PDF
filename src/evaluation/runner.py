from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.config import settings
from src.evaluation.retrieval_metrics import aggregate_scores, score_retrieval
from src.interfaces import DatabaseProtocol, EmbedderProtocol, RerankerProtocol

MODES = ["hybrid", "semantic", "keyword"]


def _gold_set_path() -> Path:
    return Path(settings.eval_gold_set_path)


def gold_set_hash(path: Path | None = None) -> str | None:
    target = path or _gold_set_path()
    if not target.exists():
        return None
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return digest[:12]


def gold_set_status() -> dict:
    path = _gold_set_path()
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "count": 0,
            "hash": None,
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        count = len(data) if isinstance(data, list) else 0
    except (json.JSONDecodeError, OSError):
        count = 0
    return {
        "path": str(path),
        "exists": True,
        "count": count,
        "hash": gold_set_hash(path),
    }


def load_qa_file(path: str | Path) -> tuple[list[dict], bool]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "questions" in raw:
        qa = []
        for item in raw["questions"]:
            keywords = item.get("expected_keywords") or []
            if isinstance(keywords, str):
                keywords = [keywords]
            qa.append(
                {
                    "question": item["question"],
                    "references": keywords,
                    "source_pdf": item.get("source_pdf"),
                }
            )
        return qa, False

    if isinstance(raw, list) and raw and "ground_truth_contexts" in raw[0]:
        qa = []
        for item in raw:
            refs = item.get("ground_truth_contexts") or []
            if isinstance(refs, str):
                refs = [refs]
            qa.append(
                {
                    "question": item["question"],
                    "references": refs,
                    "source_pdf": item.get("source_pdf"),
                    "ground_truth": item.get("ground_truth"),
                }
            )
        return qa, True

    if isinstance(raw, list):
        qa = []
        for item in raw:
            refs = item.get("references")
            if refs is None:
                answer = item.get("answer", "")
                refs = [answer] if answer else []
            qa.append(
                {
                    "question": item["question"],
                    "references": refs,
                    "source_pdf": item.get("source_pdf"),
                }
            )
        return qa, False

    raise ValueError("Unsupported QA file format")


async def _retrieve_chunks(
    question: str,
    top_k: int,
    mode: str,
    rerank: bool,
    db: DatabaseProtocol,
    embedder: EmbedderProtocol,
    reranker: RerankerProtocol,
) -> list[dict]:
    query_vector = (await embedder.embed([question]))[0]
    pool_k = top_k
    if rerank and settings.rerank_enabled:
        pool_k = min(
            top_k * settings.rerank_candidate_multiplier,
            settings.rerank_max_candidates,
        )
    chunks = db.search_chunks(
        query_vector,
        query_text=question,
        k=pool_k,
        search_mode=mode,
    )
    if rerank and settings.rerank_enabled:
        chunks = await reranker.rerank(question, chunks, top_k)
    return chunks


def _aggregate_by_source(per_question: list[dict]) -> dict[str, dict]:
    buckets: dict[str, dict[str, list[float]]] = {}
    for row in per_question:
        source = row.get("source_pdf") or "unknown"
        bucket = buckets.setdefault(
            source,
            {
                "precision": [],
                "recall": [],
                "context_precision": [],
                "context_recall": [],
            },
        )
        bucket["precision"].append(row["precision"])
        bucket["recall"].append(row["recall"])
        bucket["context_precision"].append(row["context_precision"])
        bucket["context_recall"].append(row["context_recall"])

    return {
        source: aggregate_scores(
            values["precision"],
            values["recall"],
            values["context_precision"],
            values["context_recall"],
        )
        for source, values in buckets.items()
    }


async def run_retrieval_eval(
    qa: list[dict],
    is_gold: bool,
    top_k: int,
    db: DatabaseProtocol,
    embedder: EmbedderProtocol,
    reranker: RerankerProtocol,
) -> dict:
    variants = [(mode, False) for mode in MODES]
    if settings.rerank_enabled:
        variants.extend((mode, True) for mode in MODES)

    results: dict[str, dict] = {}
    for mode, use_rerank in variants:
        precision_scores: list[float] = []
        recall_scores: list[float] = []
        context_precision_scores: list[float] = []
        context_recall_scores: list[float] = []
        per_question: list[dict] = []

        for idx, item in enumerate(qa, start=1):
            try:
                chunks = await _retrieve_chunks(
                    item["question"],
                    top_k,
                    mode,
                    use_rerank,
                    db,
                    embedder,
                    reranker,
                )
            except Exception:
                chunks = []

            precision, recall, context_precision, context_recall = score_retrieval(
                chunks,
                item["references"],
                is_gold,
            )
            precision_scores.append(precision)
            recall_scores.append(recall)
            context_precision_scores.append(context_precision)
            context_recall_scores.append(context_recall)
            per_question.append(
                {
                    "id": idx,
                    "question": item["question"],
                    "source_pdf": item.get("source_pdf"),
                    "precision": precision,
                    "recall": recall,
                    "context_precision": context_precision,
                    "context_recall": context_recall,
                    "chunks_returned": len(chunks),
                }
            )

        label = f"{mode}+rerank" if use_rerank else mode
        summary = aggregate_scores(
            precision_scores,
            recall_scores,
            context_precision_scores,
            context_recall_scores,
        )
        results[label] = {
            **summary,
            "mode": mode,
            "rerank": use_rerank,
            "per_question": per_question,
            "by_source": _aggregate_by_source(per_question),
        }

    return results
