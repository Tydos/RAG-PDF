from __future__ import annotations

import json
import logging
import re

from src.config import settings
from src.inference.llm_adapters import HuggingFaceAdapter
from src.inference.prompt_builder import PromptBuilder
from src.interfaces import DatabaseProtocol, EmbedderProtocol, GeneratorProtocol, RerankerProtocol

_FAITHFULNESS_SYSTEM = (
    "You are an evaluation assistant for a RAG system. "
    "Rate whether a generated answer is fully supported by the provided context. "
    'Return ONLY valid JSON: {"score": <0.0-1.0>, "reasoning": "<one sentence>"}. '
    "Score 1.0 means every factual claim in the answer is directly supported by the context. "
    "Score 0.0 means the answer contains claims not found in the context (hallucinations). "
    "Use partial scores for mixed cases. Do not include any other text."
)

_RELEVANCE_SYSTEM = (
    "You are an evaluation assistant for a RAG system. "
    "Rate whether a generated answer addresses the user's question. "
    'Return ONLY valid JSON: {"score": <0.0-1.0>, "reasoning": "<one sentence>"}. '
    "Score 1.0 means the answer fully and directly addresses the question. "
    "Score 0.0 means the answer is completely off-topic or does not address the question. "
    "Use partial scores for mixed cases. Do not include any other text."
)

HALLUCINATION_THRESHOLD = 0.5
MODES = ["hybrid", "semantic", "keyword"]


def judge_available() -> bool:
    return bool(settings.hf_token.strip())


def _parse_score(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    score_match = re.search(r'"score"\s*:\s*([0-9.]+)', text)
    reasoning_match = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)+)"', text)
    if score_match:
        return {
            "score": float(score_match.group(1)),
            "reasoning": reasoning_match.group(1) if reasoning_match else "",
        }
    return None


def _create_judge() -> HuggingFaceAdapter:
    return HuggingFaceAdapter(
        model=settings.hf_judge_model,
        token=settings.hf_token,
        timeout=90,
    )


async def judge_faithfulness(
    judge: HuggingFaceAdapter,
    question: str,
    context: str,
    answer: str,
) -> dict:
    try:
        text = await judge.generate(
            messages=[
                {"role": "system", "content": _FAITHFULNESS_SYSTEM},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer: {answer}",
                },
            ],
            max_tokens=settings.hf_judge_max_tokens,
            temperature=0.0,
        )
    except Exception as exc:
        logging.warning("Judge faithfulness call failed: %s", exc)
        return {"score": None, "reasoning": "api error"}

    parsed = _parse_score(text)
    if parsed is None:
        logging.warning("Could not parse faithfulness score: %r", text[:120])
        return {"score": None, "reasoning": "parse error"}
    return parsed


async def judge_answer_relevance(
    judge: HuggingFaceAdapter,
    question: str,
    answer: str,
) -> dict:
    try:
        text = await judge.generate(
            messages=[
                {"role": "system", "content": _RELEVANCE_SYSTEM},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nAnswer: {answer}",
                },
            ],
            max_tokens=settings.hf_judge_max_tokens,
            temperature=0.0,
        )
    except Exception as exc:
        logging.warning("Judge relevance call failed: %s", exc)
        return {"score": None, "reasoning": "api error"}

    parsed = _parse_score(text)
    if parsed is None:
        logging.warning("Could not parse relevance score: %r", text[:120])
        return {"score": None, "reasoning": "parse error"}
    return parsed


async def _retrieve_and_generate(
    question: str,
    top_k: int,
    mode: str,
    db: DatabaseProtocol,
    embedder: EmbedderProtocol,
    reranker: RerankerProtocol,
    generator: GeneratorProtocol,
) -> tuple[str, list[dict]]:
    query_vector = (await embedder.embed([question]))[0]
    pool_k = top_k
    if settings.rerank_enabled:
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
    if settings.rerank_enabled:
        chunks = await reranker.rerank(question, chunks, top_k)
    answer = await generator.generate(question, chunks, [])
    return answer or "", chunks


async def evaluate_answer_quality_item(
    judge: HuggingFaceAdapter,
    question: str,
    answer: str,
    chunks: list[dict],
) -> dict:
    if not answer or not chunks:
        return {
            "faithfulness_score": None,
            "faithfulness_reasoning": "no answer or no chunks",
            "relevance_score": None,
            "relevance_reasoning": "no answer or no chunks",
        }

    context = PromptBuilder.format_chunks(chunks)
    faith = await judge_faithfulness(judge, question, context, answer)
    rel = await judge_answer_relevance(judge, question, answer)
    return {
        "faithfulness_score": faith.get("score"),
        "faithfulness_reasoning": faith.get("reasoning", ""),
        "relevance_score": rel.get("score"),
        "relevance_reasoning": rel.get("reasoning", ""),
    }


async def run_answer_quality_eval(
    qa: list[dict],
    top_k: int,
    db: DatabaseProtocol,
    embedder: EmbedderProtocol,
    reranker: RerankerProtocol,
    generator: GeneratorProtocol,
) -> dict[str, list[dict]]:
    if not judge_available():
        return {}

    judge = _create_judge()
    results_by_mode: dict[str, list[dict]] = {}

    for mode in MODES:
        mode_results: list[dict] = []
        for idx, item in enumerate(qa, start=1):
            question = item["question"]
            try:
                answer, chunks = await _retrieve_and_generate(
                    question,
                    top_k,
                    mode,
                    db,
                    embedder,
                    reranker,
                    generator,
                )
            except Exception as exc:
                logging.warning("Answer-quality retrieval failed for question %d: %s", idx, exc)
                mode_results.append(
                    {
                        "id": idx,
                        "question": question,
                        "source_pdf": item.get("source_pdf"),
                        "answer": None,
                        "chunks_returned": 0,
                        "faithfulness_score": None,
                        "faithfulness_reasoning": "api error",
                        "relevance_score": None,
                        "relevance_reasoning": "api error",
                    }
                )
                continue

            scores = await evaluate_answer_quality_item(judge, question, answer, chunks)
            mode_results.append(
                {
                    "id": idx,
                    "question": question,
                    "source_pdf": item.get("source_pdf"),
                    "answer": answer,
                    "chunks_returned": len(chunks),
                    **scores,
                }
            )
        results_by_mode[mode] = mode_results

    return results_by_mode


def summarize_answer_quality(results_by_mode: dict[str, list[dict]]) -> dict:
    summary: dict[str, dict] = {}
    for mode, results in results_by_mode.items():
        scored = [
            row
            for row in results
            if row.get("faithfulness_score") is not None and row.get("relevance_score") is not None
        ]
        if not scored:
            summary[mode] = {
                "faithfulness": None,
                "relevance": None,
                "hallucination_rate": None,
                "sample_count": 0,
                "skipped": len(results),
                "by_source": {},
            }
            continue

        faithfulness = sum(row["faithfulness_score"] for row in scored) / len(scored)
        relevance = sum(row["relevance_score"] for row in scored) / len(scored)
        hallucinations = sum(
            1 for row in scored if row["faithfulness_score"] < HALLUCINATION_THRESHOLD
        )
        by_source: dict[str, dict] = {}
        for row in scored:
            source = row.get("source_pdf") or "unknown"
            bucket = by_source.setdefault(
                source,
                {"faithfulness": [], "relevance": [], "hallucinations": 0, "count": 0},
            )
            bucket["faithfulness"].append(row["faithfulness_score"])
            bucket["relevance"].append(row["relevance_score"])
            bucket["count"] += 1
            if row["faithfulness_score"] < HALLUCINATION_THRESHOLD:
                bucket["hallucinations"] += 1

        summary[mode] = {
            "faithfulness": faithfulness,
            "relevance": relevance,
            "hallucination_rate": hallucinations / len(scored),
            "sample_count": len(scored),
            "skipped": len(results) - len(scored),
            "by_source": {
                source: {
                    "faithfulness": sum(values["faithfulness"]) / len(values["faithfulness"]),
                    "relevance": sum(values["relevance"]) / len(values["relevance"]),
                    "hallucination_rate": values["hallucinations"] / values["count"],
                    "sample_count": values["count"],
                }
                for source, values in by_source.items()
            },
        }
    return summary
