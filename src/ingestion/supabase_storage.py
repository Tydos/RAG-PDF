from urllib.parse import quote

import httpx

from src.config import settings


def _object_url(filename: str) -> str:
    encoded = "/".join(quote(part, safe="") for part in filename.split("/"))
    return f"{settings.supabase_url}/storage/v1/object/{settings.supabase_bucket}/{encoded}"


async def upload_to_supabase(filename: str, data: bytes, content_type: str) -> str:
    if not settings.supabase_service_key:
        raise RuntimeError("SUPABASE_SERVICE_KEY is not configured.")
    headers = {
        "Authorization": f"Bearer {settings.supabase_service_key}",
        "Content-Type": content_type,
        "x-upsert": "true",
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(_object_url(filename), content=data, headers=headers)
        resp.raise_for_status()
    encoded = "/".join(quote(part, safe="") for part in filename.split("/"))
    return (
        f"{settings.supabase_url}/storage/v1/object/public"
        f"/{settings.supabase_bucket}/{encoded}"
    )


async def delete_from_supabase(filename: str) -> None:
    if not settings.supabase_service_key:
        raise RuntimeError("SUPABASE_SERVICE_KEY is not configured.")
    headers = {"Authorization": f"Bearer {settings.supabase_service_key}"}
    async with httpx.AsyncClient() as client:
        resp = await client.delete(_object_url(filename), headers=headers)
        if resp.status_code not in (200, 204, 404):
            resp.raise_for_status()
