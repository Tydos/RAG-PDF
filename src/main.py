import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.config import PROJECT_ROOT, settings
from src.inference.embeddings import HuggingFaceEmbeddingService
from src.inference.rag_generator import create_rag_generator
from src.inference.reranker import HuggingFaceReranker
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.service import IngestionService
from src.routes import chat, documents, eval, health, pages
from src.storage.database import DBManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s",
    stream=sys.stdout,
)


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

if settings.cors_origin_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")

app.include_router(pages.router)
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(eval.router)
