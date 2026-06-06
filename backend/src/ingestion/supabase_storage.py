import httpx

from src.config import settings


async def upload_to_supabase(filename: str, data: bytes, content_type: str) -> str:
    if not settings.supabase_service_key:
        raise RuntimeError("SUPABASE_SERVICE_KEY is not configured.")
    url = f"{settings.supabase_url}/storage/v1/object/{settings.supabase_bucket}/{filename}"
    headers = {
        "Authorization": f"Bearer {settings.supabase_service_key}",
        "Content-Type": content_type,
        "x-upsert": "true",
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, content=data, headers=headers)
        resp.raise_for_status()
    return (
        f"{settings.supabase_url}/storage/v1/object/public"
        f"/{settings.supabase_bucket}/{filename}"
    )
