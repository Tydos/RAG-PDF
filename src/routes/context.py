from fastapi import Request
from fastapi.templating import Jinja2Templates

from src.config import PROJECT_ROOT, settings
from src.evaluation.answer_quality import judge_available
from src.evaluation.runner import gold_set_status
from src.interfaces import DatabaseProtocol

templates = Jinja2Templates(directory=str(PROJECT_ROOT / "src" / "templates"))


def is_htmx(request: Request) -> bool:
    return request.headers.get("HX-Request") == "true"


def blob_by_filename(documents: list[dict]) -> dict[str, str]:
    return {d["filename"]: d.get("blob_url") or "" for d in documents}


def documents_context(db: DatabaseProtocol) -> dict:
    documents = db.list_uploads()
    return {
        "documents": documents,
        "has_pending": any(d.get("status") == "pending" for d in documents),
    }


def eval_context(db: DatabaseProtocol) -> dict:
    latest_run = db.get_latest_eval_run()
    gold = gold_set_status()
    return {
        "latest_run": latest_run,
        "gold_set": gold,
        "judge_available": judge_available(),
        "rerank_enabled": settings.rerank_enabled,
        "runs": db.list_eval_runs(limit=settings.eval_runs_partial_limit),
        "judge_model": settings.hf_judge_model,
    }
