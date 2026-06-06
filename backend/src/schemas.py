from typing import Literal
from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    filenames: list[str] | None = None
    search_mode: Literal["hybrid", "semantic", "keyword"] = "hybrid"


ChatRequest = QueryRequest


class IngestRequest(BaseModel):
    filename: str
    url: str
