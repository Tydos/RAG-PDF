import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, File, Request, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from src.config import settings
from src.interfaces import DatabaseProtocol, EmbedderProtocol, GeneratorProtocol
from src.schemas import ChatRequest, IngestRequest, QueryRequest
from src.inference.rag_generator import create_rag_generator
from src.inference.embeddings import HuggingFaceEmbeddingService
from src.storage.database import DBManager
from src.ingestion.service import IngestionService
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.supabase_storage import upload_to_supabase
from src.utils.latency import LatencyTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s",
    stream=sys.stdout,
)

# --- App startup ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = DBManager.create()
    app.state.embedder = HuggingFaceEmbeddingService()
    app.state.generator = create_rag_generator()
    app.state.extractor = PDFParser()
    app.state.pipeline = IngestionService(app.state.db, app.state.embedder, app.state.extractor)
    yield


app = FastAPI(title="Antares API", lifespan=lifespan)
router = APIRouter()


# --- Dependency helpers ---


def get_db(request: Request) -> DatabaseProtocol:
    return request.app.state.db


def get_embedder(request: Request) -> EmbedderProtocol:
    return request.app.state.embedder


def get_generator(request: Request) -> GeneratorProtocol:
    return request.app.state.generator


def get_pipeline(request: Request) -> IngestionService:
    return request.app.state.pipeline


# --- Routes ---


@router.get("/")
def root():
    return {"message": "Antares API. Visit /docs for the interactive API explorer."}


@router.get("/health")
def health(db: DBManager = Depends(get_db)):
    db_ok = db.ping()
    return JSONResponse(
        {"status": "ok" if db_ok else "degraded", "services": {"database": db_ok}},
        status_code=200 if db_ok else 503,
    )


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: DBManager = Depends(get_db),
    pipeline: IngestionService = Depends(get_pipeline),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    data = await file.read()
    try:
        public_url = await upload_to_supabase(
            file.filename, data, file.content_type or "application/pdf"
        )
    except Exception as e:
        logging.exception("Supabase upload failed")
        raise HTTPException(status_code=500, detail=f"Storage upload failed: {e}") from e
    db.add_upload(file.filename, public_url)
    background_tasks.add_task(pipeline.index_document, file.filename, public_url)
    return {"status": "upload recorded, indexing in progress", "url": public_url}


@router.post("/ingest")
async def ingest_document(
    req: IngestRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: DBManager = Depends(get_db),
    pipeline: IngestionService = Depends(get_pipeline),
):
    if not req.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    db.add_upload(req.filename, req.url)
    background_tasks.add_task(pipeline.index_document, req.filename, req.url)
    return {"status": "upload recorded, indexing in progress", "url": req.url}


@router.get("/documents")
def list_documents(db: DBManager = Depends(get_db)):
    return {"documents": db.list_uploads()}


@router.post("/query")
async def query(
    req: QueryRequest,
    db: DatabaseProtocol = Depends(get_db),
    embedder: EmbedderProtocol = Depends(get_embedder),
    generator: GeneratorProtocol = Depends(get_generator),
):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    top_k = max(1, min(req.top_k, 20))
    tracker = LatencyTracker()

    try:
        with tracker.measure("embed"):
            query_vector = (await embedder.embed([question]))[0]
    except Exception as e:
        logging.exception("Embedding failed")
        raise HTTPException(status_code=503, detail=f"Embedding service error: {e}")

    with tracker.measure("search"):
        chunks = db.search_chunks(
            query_vector,
            query_text=question,
            k=top_k,
            filenames=req.filenames or None,
            search_mode=req.search_mode,
        )

    answer: str | None = None
    try:
        with tracker.measure("llm"):
            answer = await generator.generate(question, chunks, [])
    except Exception:
        logging.exception("Answer generation failed; returning raw chunks")

    latency = tracker.all()
    logging.info(
        "query latency — embed: %.0fms | search: %.0fms | llm: %.0fms | total: %.0fms | chunks: %d",
        latency.get("embed", 0),
        latency.get("search", 0),
        latency.get("llm", 0),
        latency.get("total", 0),
        len(chunks),
    )

    return {
        "question": question,
        "answer": answer,
        "chunks": chunks,
        "latency": latency,
    }


@router.get("/history")
def get_history(db: DBManager = Depends(get_db)):
    return {"messages": db.get_messages()}


@router.post("/chat")
async def chat(
    req: ChatRequest,
    db: DBManager = Depends(get_db),
    embedder: EmbedderProtocol = Depends(get_embedder),
    generator: GeneratorProtocol = Depends(get_generator),
):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    top_k = max(1, min(req.top_k, 20))
    tracker = LatencyTracker()

    try:
        with tracker.measure("embed"):
            query_vector = (await embedder.embed([question]))[0]
    except Exception as e:
        logging.exception("Embedding failed")
        raise HTTPException(status_code=503, detail=f"Embedding service error: {e}")

    with tracker.measure("search"):
        chunks = db.search_chunks(
            query_vector,
            query_text=question,
            k=top_k,
            filenames=req.filenames or None,
            search_mode=req.search_mode,
        )

    history = db.get_messages()

    answer: str | None = None
    try:
        with tracker.measure("llm"):
            answer = await generator.generate(question, chunks, history)
    except Exception:
        logging.exception("Answer generation failed; returning raw chunks")

    db.add_message("user", question)
    db.add_message("assistant", answer or "", chunks)

    latency = tracker.all()
    logging.info(
        "chat latency — embed: %.0fms | search: %.0fms | llm: %.0fms | total: %.0fms | chunks: %d",
        latency.get("embed", 0),
        latency.get("search", 0),
        latency.get("llm", 0),
        latency.get("total", 0),
        len(chunks),
    )

    return {
        "answer": answer,
        "chunks": chunks,
        "latency": latency,
    }


@router.get("/eval/summary")
def eval_summary():
    eval_dir = Path(__file__).resolve().parent.parent / "tests" / "retriever-evaluation"

    retrieval = None
    ret_path = eval_dir / "results.json"
    if ret_path.exists():
        with open(ret_path) as f:
            raw = json.load(f)
        retrieval = {}
        for mode, data in raw.items():
            retrieval[mode] = {
                "precision": data.get("precision"),
                "recall": data.get("recall"),
                "f1": data.get("f1"),
                "n": len(data.get("per_question", [])),
            }

    answer_quality = None
    aq_path = eval_dir / "aq_results.json"
    if aq_path.exists():
        with open(aq_path) as f:
            raw = json.load(f)
        answer_quality = {}
        for mode, questions in raw.items():
            faith_scores = [
                q["faithfulness_score"]
                for q in questions
                if q.get("faithfulness_score") is not None
            ]
            rel_scores = [
                q["relevance_score"] for q in questions if q.get("relevance_score") is not None
            ]
            n = len(faith_scores)
            avg_faith = sum(faith_scores) / n if n else None
            avg_rel = sum(rel_scores) / len(rel_scores) if rel_scores else None
            answer_quality[mode] = {
                "avg_faithfulness": round(avg_faith, 4) if avg_faith is not None else None,
                "avg_relevance": round(avg_rel, 4) if avg_rel is not None else None,
                "hallucination_rate": round(1 - avg_faith, 4) if avg_faith is not None else None,
                "n": n,
            }

    return {"retrieval": retrieval, "answer_quality": answer_quality}


@router.delete("/files/{filename}")
def delete_file(filename: str, db: DBManager = Depends(get_db)):
    try:
        db.remove_upload(filename)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"deleted": filename}


app.include_router(router)
