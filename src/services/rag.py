import logging

from fastapi.exceptions import HTTPException

from src.config import settings
from src.inference.llm_adapters import LLMAuthError
from src.interfaces import (
    DatabaseProtocol,
    EmbedderProtocol,
    GeneratorProtocol,
    RerankerProtocol,
)
from src.utils.latency import LatencyTracker


async def run_rag(
    question: str,
    top_k: int,
    filenames: list[str] | None,
    search_mode: str,
    persist: bool,
    rerank: bool,
    db: DatabaseProtocol,
    embedder: EmbedderProtocol,
    generator: GeneratorProtocol,
    reranker: RerankerProtocol,
) -> dict:
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    top_k = max(1, min(top_k, settings.query_top_k_max))
    use_rerank = rerank and settings.rerank_enabled
    tracker = LatencyTracker()

    try:
        with tracker.measure("embed"):
            query_vector = (await embedder.embed([question]))[0]
    except Exception as e:
        logging.exception("Embedding failed")
        raise HTTPException(status_code=503, detail=f"Embedding service error: {e}") from e

    pool_k = top_k
    if use_rerank:
        pool_k = min(
            top_k * settings.rerank_candidate_multiplier,
            settings.rerank_max_candidates,
        )

    with tracker.measure("search"):
        chunks = db.search_chunks(
            query_vector,
            query_text=question,
            k=pool_k,
            filenames=filenames or None,
            search_mode=search_mode,
        )

    if use_rerank:
        with tracker.measure("rerank"):
            chunks = await reranker.rerank(question, chunks, top_k)

    history = db.get_messages() if persist else []

    answer: str | None = None
    try:
        with tracker.measure("llm"):
            answer = await generator.generate(question, chunks, history)
    except LLMAuthError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e
    except Exception:
        logging.exception("Answer generation failed; returning raw chunks")

    if persist:
        db.add_message("user", question)
        db.add_message("assistant", answer or "", chunks)

    latency = tracker.all()
    logging.info(
        "chat latency — embed: %.0fms | search: %.0fms | rerank: %.0fms | llm: %.0fms | total: %.0fms | chunks: %d | persist: %s | rerank: %s",
        latency.get("embed", 0),
        latency.get("search", 0),
        latency.get("rerank", 0),
        latency.get("llm", 0),
        latency.get("total", 0),
        len(chunks),
        persist,
        use_rerank,
    )

    return {
        "question": question,
        "answer": answer,
        "chunks": chunks,
        "latency": latency,
    }
