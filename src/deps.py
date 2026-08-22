from fastapi import Request

from src.ingestion.service import IngestionService
from src.interfaces import (
    DatabaseProtocol,
    EmbedderProtocol,
    GeneratorProtocol,
    RerankerProtocol,
)


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
