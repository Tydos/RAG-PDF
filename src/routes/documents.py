import logging

from fastapi import APIRouter, BackgroundTasks, Depends, File, Request, UploadFile
from fastapi.exceptions import HTTPException

from src.config import settings
from src.deps import get_db, get_pipeline
from src.ingestion.service import IngestionService
from src.ingestion.storage import delete_file as delete_stored_file
from src.ingestion.storage import upload_file as store_file
from src.routes.context import documents_context, is_htmx, templates
from src.storage.database import DBManager

router = APIRouter(tags=["documents"])


@router.post("/upload")
async def upload_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: DBManager = Depends(get_db),
    pipeline: IngestionService = Depends(get_pipeline),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        limit_mb = settings.max_upload_bytes // (1024 * 1024)
        raise HTTPException(status_code=400, detail=f"Exceeds the {limit_mb} MB limit.")
    try:
        public_url = await store_file(file.filename, data, file.content_type or "application/pdf")
    except Exception as e:
        logging.exception("File upload failed")
        raise HTTPException(status_code=500, detail=f"Storage upload failed: {e}") from e
    db.add_upload(file.filename, public_url)
    background_tasks.add_task(pipeline.index_document, file.filename, public_url)

    if is_htmx(request):
        return templates.TemplateResponse(
            request,
            "partials/documents.html",
            documents_context(db),
        )
    return {"status": "upload recorded, indexing in progress", "url": public_url}


@router.get("/documents")
def list_documents(db: DBManager = Depends(get_db)):
    return {"documents": db.list_uploads()}


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

    if is_htmx(request):
        return templates.TemplateResponse(
            request,
            "partials/documents.html",
            documents_context(db),
        )
    return {"deleted": filename}
