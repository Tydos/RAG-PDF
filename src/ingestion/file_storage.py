"""File storage — local disk when Supabase is not configured, otherwise Supabase."""

import asyncio
import logging
from pathlib import Path
from urllib.parse import quote

from src.config import settings
from src.ingestion import local_storage, supabase_storage


def uses_local_storage() -> bool:
    return not settings.supabase_service_key.strip()


async def store_file(filename: str, data: bytes, content_type: str) -> str:
    if uses_local_storage():
        logging.info("Using local file storage (SUPABASE_SERVICE_KEY not set)")
        return await asyncio.to_thread(local_storage.save, filename, data)
    return await supabase_storage.upload_to_supabase(filename, data, content_type)


async def delete_file(filename: str) -> None:
    if uses_local_storage():
        await asyncio.to_thread(local_storage.delete, filename)
        return
    await supabase_storage.delete_from_supabase(filename)


def local_upload_dir() -> Path:
    return local_storage.upload_dir()


def mount_uploads_if_local(app) -> None:
    if uses_local_storage():
        upload_dir = local_upload_dir()
        upload_dir.mkdir(parents=True, exist_ok=True)
        app.mount("/uploads", StaticFiles(directory=str(upload_dir)), name="uploads")
