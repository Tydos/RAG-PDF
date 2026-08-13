from typing import Literal

from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5
    filenames: list[str] | None = None
    search_mode: Literal["hybrid", "semantic", "keyword"] = "hybrid"
    persist: bool = True
    rerank: bool = True
