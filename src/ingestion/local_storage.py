"""Local disk storage for development (no Supabase required)."""

from pathlib import Path

from src.config import settings

BASE_DIR = Path(__file__).resolve().parents[2]


def upload_dir() -> Path:
    path = Path(settings.local_upload_dir)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def _safe_path(filename: str) -> Path:
    name = Path(filename).name
    if not name or name in (".", ".."):
        raise ValueError("Invalid filename.")
    return upload_dir() / name


def save(filename: str, data: bytes) -> str:
    dest = _safe_path(filename)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    encoded = quote(Path(filename).name, safe="")
    return f"{settings.app_base_url.rstrip('/')}/uploads/{encoded}"


def delete(filename: str) -> None:
    path = _safe_path(filename)
    if path.exists():
        path.unlink()


def resolve_path(blob_url: str, filename: str) -> Path | None:
    """Return a local path if blob_url points at this app's /uploads/ serving."""
    prefix = f"{settings.app_base_url.rstrip('/')}/uploads/"
    if blob_url.startswith(prefix):
        name = blob_url[len(prefix) :].split("?", 1)[0]
        return _safe_path(name)
    if blob_url.startswith("/uploads/"):
        return _safe_path(blob_url[len("/uploads/") :])
    return None
