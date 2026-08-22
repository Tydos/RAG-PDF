from fastapi import APIRouter, Depends, Form, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from src.config import settings
from src.deps import get_db, get_embedder, get_reranker
from src.evaluation.answer_quality import (
    judge_available,
    run_answer_quality_eval,
    summarize_answer_quality,
)
from src.evaluation.gold_set import generate_gold_set
from src.evaluation.runner import gold_set_hash, gold_set_status, load_qa_file, run_retrieval_eval
from src.inference.rag_generator import create_rag_generator
from src.interfaces import EmbedderProtocol, RerankerProtocol
from src.routes.context import eval_context, is_htmx, templates
from src.storage.database import DBManager

router = APIRouter(tags=["eval"])


@router.get("/eval/summary")
def eval_summary(db: DBManager = Depends(get_db)):
    latest = db.get_latest_eval_run()
    if not latest:
        return JSONResponse({"status": "empty", "run": None})
    return {
        "status": "ok",
        "run": latest,
        "gold_set": gold_set_status(),
        "judge_available": judge_available(),
    }


@router.post("/eval/gold-set/generate")
async def eval_generate_gold_set(
    request: Request,
    sample_n: int = Form(20),
    db: DBManager = Depends(get_db),
):
    if not settings.hf_token:
        raise HTTPException(
            status_code=400,
            detail="HF_TOKEN is required to generate a gold set with the HF judge model.",
        )

    sample_n = max(0, min(sample_n, settings.gold_set_sample_max))
    try:
        result = await generate_gold_set(sample_n, db)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if is_htmx(request):
        return templates.TemplateResponse(
            request,
            "partials/eval_results.html",
            {**eval_context(db), "gold_set_generation": result},
        )
    return result


@router.post("/eval/run")
async def eval_run(
    request: Request,
    top_k: int = Form(5),
    include_answer_quality: bool = Form(False),
    llm: str = Form("auto"),
    db: DBManager = Depends(get_db),
    embedder: EmbedderProtocol = Depends(get_embedder),
    reranker: RerankerProtocol = Depends(get_reranker),
):
    gold = gold_set_status()
    if not gold["exists"] or gold["count"] == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No gold set found at {gold['path']}. "
                "Generate one with: python tests/retriever-evaluation/generate_gold_set.py"
            ),
        )

    qa, is_gold = load_qa_file(gold["path"])
    top_k = max(1, min(top_k, settings.query_top_k_max))
    run_answer_quality = include_answer_quality and judge_available()
    eval_generator = create_rag_generator(llm)
    resolved_llm = getattr(eval_generator, "resolved_llm", llm or "auto")

    retrieval_results = await run_retrieval_eval(
        qa,
        is_gold,
        top_k,
        db,
        embedder,
        reranker,
    )

    answer_quality_results = None
    answer_quality_summary = None
    if run_answer_quality:
        mode_results = await run_answer_quality_eval(
            qa,
            top_k,
            db,
            embedder,
            reranker,
            eval_generator,
        )
        if mode_results:
            answer_quality_summary = summarize_answer_quality(mode_results)
            answer_quality_results = {
                "results": mode_results,
                "summary": answer_quality_summary,
            }

    config = {
        "gold_set_path": gold["path"],
        "gold_set_hash": gold.get("hash") or gold_set_hash(),
        "question_count": len(qa),
        "top_k": top_k,
        "is_gold": is_gold,
        "include_answer_quality": run_answer_quality,
        "rerank_enabled": settings.rerank_enabled,
        "llm": resolved_llm,
        "judge_model": settings.hf_judge_model,
        "embed_model": settings.hf_embed_model,
        "chunk_size": settings.pdf_chunk_size,
        "chunk_overlap": settings.pdf_chunk_overlap,
    }
    run_id = db.save_eval_run(
        config,
        retrieval_results,
        answer_quality_results,
    )

    if is_htmx(request):
        return templates.TemplateResponse(
            request,
            "partials/eval_results.html",
            eval_context(db),
        )

    latest = db.get_eval_run(run_id)
    return latest or {"id": run_id, "status": "saved"}


@router.get("/eval/{run_id}", response_class=HTMLResponse)
def eval_run_detail(run_id: int, request: Request, db: DBManager = Depends(get_db)):
    run = db.get_eval_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return templates.TemplateResponse(
        request,
        "eval_run_detail.html",
        {"run": run},
    )
