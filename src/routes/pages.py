from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from src.config import settings
from src.deps import get_db
from src.routes.context import (
    blob_by_filename,
    documents_context,
    eval_context,
    templates,
)
from src.storage.database import DBManager

router = APIRouter(tags=["pages"])


@router.get("/", response_class=HTMLResponse)
def chat_page(request: Request, db: DBManager = Depends(get_db)):
    documents = db.list_uploads()
    messages = db.get_messages()
    ctx = documents_context(db)
    return templates.TemplateResponse(
        request,
        "chat.html",
        {
            "messages": messages,
            "blob_by_filename": blob_by_filename(documents),
            **ctx,
        },
    )


@router.get("/eval", response_class=HTMLResponse)
def eval_page(request: Request, db: DBManager = Depends(get_db)):
    return templates.TemplateResponse(
        request,
        "eval.html",
        eval_context(db),
    )


@router.get("/partials/documents", response_class=HTMLResponse)
def documents_partial(request: Request, db: DBManager = Depends(get_db)):
    return templates.TemplateResponse(
        request,
        "partials/documents.html",
        documents_context(db),
    )


@router.get("/partials/eval-results", response_class=HTMLResponse)
def eval_results_partial(request: Request, db: DBManager = Depends(get_db)):
    return templates.TemplateResponse(
        request,
        "partials/eval_results.html",
        eval_context(db),
    )


@router.get("/partials/eval-history", response_class=HTMLResponse)
def eval_history_partial(request: Request, db: DBManager = Depends(get_db)):
    return templates.TemplateResponse(
        request,
        "partials/eval_history.html",
        {"runs": db.list_eval_runs(limit=settings.eval_runs_partial_limit)},
    )


@router.get("/eval/history", response_class=HTMLResponse)
def eval_history_page(request: Request, db: DBManager = Depends(get_db)):
    return templates.TemplateResponse(
        request,
        "partials/eval_history.html",
        {"runs": db.list_eval_runs(limit=settings.eval_runs_history_limit)},
    )
