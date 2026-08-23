import asyncio
from urllib.parse import quote

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from src.config import settings


def _client():
    if not settings.s3_endpoint.strip():
        raise RuntimeError("S3_ENDPOINT is not configured.")
    if not settings.s3_access_key.strip() or not settings.s3_secret_key.strip():
        raise RuntimeError("S3_ACCESS_KEY and S3_SECRET_KEY are required.")
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint.rstrip("/"),
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        ),
    )


def _public_url(filename: str) -> str:
    if not settings.s3_public_url.strip():
        raise RuntimeError("S3_PUBLIC_URL is not configured.")
    encoded = "/".join(quote(part, safe="") for part in filename.split("/"))
    return f"{settings.s3_public_url.rstrip('/')}/{settings.s3_bucket}/{encoded}"


async def upload_file(filename: str, data: bytes, content_type: str) -> str:
    client = _client()
    await asyncio.to_thread(
        client.put_object,
        Bucket=settings.s3_bucket,
        Key=filename,
        Body=data,
        ContentType=content_type,
    )
    return _public_url(filename)


async def delete_file(filename: str) -> None:
    client = _client()
    try:
        await asyncio.to_thread(
            client.delete_object,
            Bucket=settings.s3_bucket,
            Key=filename,
        )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404", "NotFound"):
            return
        raise
