from src.config import settings
from src.ingestion.s3_storage import delete_from_s3, upload_to_s3
from src.ingestion.supabase_storage import delete_from_supabase, upload_to_supabase


async def upload_file(filename: str, data: bytes, content_type: str) -> str:
    if settings.use_s3_storage:
        return await upload_to_s3(filename, data, content_type)
    return await upload_to_supabase(filename, data, content_type)


async def delete_file(filename: str) -> None:
    if settings.use_s3_storage:
        await delete_from_s3(filename)
    else:
        await delete_from_supabase(filename)
