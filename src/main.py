import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Request,
    UploadFile,
)
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import settings
from src.evaluation.answer_quality import (
    judge_available,
    run_answer_quality_eval,
    summarize_answer_quality,
)
from src.evaluation.gold_set import generate_gold_set
from src.evaluation.runner import gold_set_hash, gold_set_status, load_qa_file, run_retrieval_eval
from src.inference.embeddings import HuggingFaceEmbeddingService
from src.inference.llm_adapters import LLMAuthError
from src.inference.rag_generator import create_rag_generator
from src.inference.reranker import HuggingFaceReranker
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.service import IngestionService
from src.ingestion.storage import delete_file as delete_stored_file
from src.ingestion.storage import upload_file as store_file
from src.interfaces import (
    DatabaseProtocol,
    EmbedderProtocol,
    GeneratorProtocol,
    RerankerProtocol,
)
from src.schemas import ChatRequest
from src.storage.database import DBManager
from src.utils.latency import LatencyTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s",
    stream=sys.stdout,
)

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "src" / "templates"))

MAX_UPLOAD_BYTES = 100 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = DBManager.create()
    app.state.embedder = HuggingFaceEmbeddingService()
    app.state.generator = create_rag_generator()
    app.state.reranker = HuggingFaceReranker()
    app.state.extractor = PDFParser()
    app.state.pipeline = IngestionService(app.state.db, app.state.embedder, app.state.extractor)
    yield


app = FastAPI(
    title="Antares",
    summary="RAG retrieval evaluation platform with hybrid search, reranking, and answer-quality metrics.",
    lifespan=lifespan,
)

_cors = os.getenv("CORS_ORIGINS", "").strip()
if _cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in _cors.split(",") if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

router = APIRouter()


def get_db(request: Request) -> DatabaseProtocol:
    return request.app.state.db


def get_embedder(request: Request) -> EmbedderProtocol:
    return request.app.state.embedder


def get_generator(request: Request) -> GeneratorProtocol:
    return request.app.state.generator


def get_reranker(request: Request) -> RerankerProtocol:
    return request.app.state.reranker


def get_pipeline(request: Request) -> IngestionService:
    return request.app.state.pipeline


def _is_htmx(request: Request) -> bool:
    return request.headers.get("HX-Request") == "true"


def _blob_by_filename(documents: list[dict]) -> dict[str, str]:
    return {d["filename"]: d.get("blob_url") or "" for d in documents}


def _documents_context(db: DatabaseProtocol) -> dict:
    documents = db.list_uploads()
    return {
        "documents": documents,
        "has_pending": any(d.get("status") == "pending" for d in documents),
    }


def _eval_context(db: DatabaseProtocol) -> dict:
    latest_run = db.get_latest_eval_run()
    gold = gold_set_status()
    return {
        "latest_run": latest_run,
        "gold_set": gold,
        "judge_available": judge_available(),
        "rerank_enabled": settings.rerank_enabled,
        "runs": db.list_eval_runs(limit=20),
        "judge_model": settings.hf_judge_model,
    }


async def _run_rag(
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


# --- HTML pages ---


@router.get("/", response_class=HTMLResponse)
def chat_page(request: Request, db: DBManager = Depends(get_db)):
    documents = db.list_uploads()
    messages = db.get_messages()
    ctx = _documents_context(db)
    return templates.TemplateResponse(
        request,
        "chat.html",
        {
            "messages": messages,
            "blob_by_filename": _blob_by_filename(documents),
            **ctx,
        },
    )


@router.get("/eval", response_class=HTMLResponse)
def eval_page(request: Request, db: DBManager = Depends(get_db)):
    return templates.TemplateResponse(
        request,
        "eval.html",
        _eval_context(db),
    )


@router.get("/partials/documents", response_class=HTMLResponse)
def documents_partial(request: Request, db: DBManager = Depends(get_db)):
    return templates.TemplateResponse(
        request,
        "partials/documents.html",
        _documents_context(db),
    )


@router.get("/partials/eval-results", response_class=HTMLResponse)
def eval_results_partial(request: Request, db: DBManager = Depends(get_db)):
    return templates.TemplateResponse(
        request,
        "partials/eval_results.html",
        _eval_context(db),
    )


@router.get("/partials/eval-history", response_class=HTMLResponse)
def eval_history_partial(request: Request, db: DBManager = Depends(get_db)):
    return templates.TemplateResponse(
        request,
        "partials/eval_history.html",
        {"runs": db.list_eval_runs(limit=20)},
    )


@router.get("/eval/history", response_class=HTMLResponse)
def eval_history_page(request: Request, db: DBManager = Depends(get_db)):
    return templates.TemplateResponse(
        request,
        "partials/eval_history.html",
        {"runs": db.list_eval_runs(limit=50)},
    )


# --- JSON API ---


@router.get("/health")
def health(db: DBManager = Depends(get_db)):
    db_ok = db.ping()
    return JSONResponse(
        {"status": "ok" if db_ok else "degraded", "services": {"database": db_ok}},
        status_code=200 if db_ok else 503,
    )


@router.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: DBManager = Depends(get_db),
    pipeline: IngestionService = Depends(get_pipeline),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Exceeds the 100 MB limit.")
    try:
        public_url = await store_file(
            file.filename, data, file.content_type or "application/pdf"
        )
    except Exception as e:
        logging.exception("File upload failed")
        raise HTTPException(status_code=500, detail=f"Storage upload failed: {e}") from e
    db.add_upload(file.filename, public_url)
    background_tasks.add_task(pipeline.index_document, file.filename, public_url)

    if _is_htmx(request):
        return templates.TemplateResponse(
            request,
            "partials/documents.html",
            _documents_context(db),
        )
    return {"status": "upload recorded, indexing in progress", "url": public_url}


@router.get("/documents")
def list_documents(db: DBManager = Depends(get_db)):
    return {"documents": db.list_uploads()}


@router.get("/history")
def get_history(db: DBManager = Depends(get_db)):
    return {"messages": db.get_messages()}


@router.post("/chat")
async def chat(
    request: Request,
    db: DatabaseProtocol = Depends(get_db),
    embedder: EmbedderProtocol = Depends(get_embedder),
    generator: GeneratorProtocol = Depends(get_generator),
    reranker: RerankerProtocol = Depends(get_reranker),
):
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        req = ChatRequest(**body)
        question = req.question
        top_k = req.top_k
        filenames = req.filenames
        search_mode = req.search_mode
        persist = req.persist
        rerank = req.rerank
    else:
        form = await request.form()
        question = str(form.get("question", ""))
        top_k = 5
        filenames = None
        search_mode = "hybrid"
        persist = True
        rerank = True

    result = await _run_rag(
        question=question,
        top_k=top_k,
        filenames=filenames,
        search_mode=search_mode,
        persist=persist,
        rerank=rerank,
        db=db,
        embedder=embedder,
        generator=generator,
        reranker=reranker,
    )

    if _is_htmx(request):
        documents = db.list_uploads()
        user_msg = {"role": "user", "content": result["question"], "chunks": []}
        assistant_msg = {
            "role": "assistant",
            "content": result["answer"] or "",
            "chunks": result["chunks"],
            "latency": result["latency"],
        }
        return templates.TemplateResponse(
            request,
            "partials/chat_exchange.html",
            {
                "user_msg": user_msg,
                "assistant_msg": assistant_msg,
                "blob_by_filename": _blob_by_filename(documents),
            },
        )

    return {
        "answer": result["answer"],
        "chunks": result["chunks"],
        "latency": result["latency"],
    }


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

    sample_n = max(0, min(sample_n, 500))
    try:
        result = await generate_gold_set(sample_n, db)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if _is_htmx(request):
        return templates.TemplateResponse(
            request,
            "partials/eval_results.html",
            {**_eval_context(db), "gold_set_generation": result},
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

    if _is_htmx(request):
        return templates.TemplateResponse(
            request,
            "partials/eval_results.html",
            _eval_context(db),
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


@router.delete("/files/{filename:path}")
async def delete_file(
    filename: str,
    request: Request,
    db: DBManager = Depends(get_db),
):
    try:
        db.remove_upload(filename)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    try:
        await delete_stored_file(filename)
    except Exception:
        logging.exception("Storage delete failed for %s", filename)

    if _is_htmx(request):
        return templates.TemplateResponse(
            request,
            "partials/documents.html",
            _documents_context(db),
        )
    return {"deleted": filename}


app.include_router(router)
