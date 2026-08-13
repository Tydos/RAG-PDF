import asyncio
import io
from urllib.parse import quote

from minio import Minio

from src.config import settings


def _client() -> Minio:
    endpoint = settings.s3_endpoint.replace("http://", "").replace("https://", "")
    return Minio(
        endpoint,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        secure=settings.s3_endpoint.startswith("https"),
    )


def _public_url(filename: str) -> str:
    encoded = "/".join(quote(part, safe="") for part in filename.split("/"))
    return f"{settings.s3_public_url.rstrip('/')}/{settings.s3_bucket}/{encoded}"


async def upload_to_s3(filename: str, data: bytes, content_type: str) -> str:
    client = _client()
    await asyncio.to_thread(
        client.put_object,
        settings.s3_bucket,
        filename,
        io.BytesIO(data),
        len(data),
        content_type=content_type,
    )
    return _public_url(filename)


async def delete_from_s3(filename: str) -> None:
    client = _client()
    try:
        await asyncio.to_thread(client.remove_object, settings.s3_bucket, filename)
    except Exception as exc:
        if "NoSuchKey" in str(exc) or "not found" in str(exc).lower():
            return
        raise
