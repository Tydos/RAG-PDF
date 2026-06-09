"""
UploadStore manages the 'uploads' table, which tracks uploaded PDF files and their processing status.
add_upload inserts a new upload record or updates an existing one with the given filename and blob
set_status updates the processing status and page count for a given filename.
remove_upload deletes an upload record by filename.
list_uploads retrieves all upload records, including their associated chunk counts, and returns them in chronological order.
"""

from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row

from src.storage.table_definitions import _CREATE_UPLOADS_TABLE


class UploadStore:
    def __init__(self, conninfo: str) -> None:
        self._conninfo = conninfo
        self._create_tables()

    def _create_tables(self) -> None:
        with psycopg.connect(self._conninfo) as conn:
            conn.execute(_CREATE_UPLOADS_TABLE)
            conn.commit()

    def add_upload(self, filename: str, blob_url: str) -> None:
        with psycopg.connect(self._conninfo) as conn:
            conn.execute(
                """
                INSERT INTO uploads (filename, blob_url, uploaded_at, status, page_count)
                VALUES (%s, %s, %s, 'pending', 0)
                ON CONFLICT (filename) DO UPDATE SET
                    blob_url = EXCLUDED.blob_url,
                    uploaded_at = EXCLUDED.uploaded_at,
                    status = 'pending',
                    page_count = 0
                """,
                (filename, blob_url, datetime.now(timezone.utc)),
            )
            conn.commit()

    def set_status(self, filename: str, status: str, page_count: int = 0) -> None:
        with psycopg.connect(self._conninfo) as conn:
            conn.execute(
                "UPDATE uploads SET status = %s, page_count = %s WHERE filename = %s",
                (status, page_count, filename),
            )
            conn.commit()

    def remove_upload(self, filename: str) -> None:
        with psycopg.connect(self._conninfo) as conn:
            cur = conn.execute(
                "DELETE FROM uploads WHERE filename = %s RETURNING filename", (filename,)
            )
            deleted = cur.fetchone()
            conn.commit()
        if not deleted:
            raise FileNotFoundError(filename)

    def list_uploads(self) -> list[dict]:
        sql = """
            SELECT u.filename, u.blob_url, u.uploaded_at, u.status, u.page_count,
                   COALESCE(c.chunk_count, 0) AS chunk_count
            FROM uploads u
            LEFT JOIN (
                SELECT filename, COUNT(*) AS chunk_count FROM chunks GROUP BY filename
            ) c ON c.filename = u.filename
            ORDER BY u.uploaded_at DESC NULLS LAST
        """
        with psycopg.connect(self._conninfo) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql)
                rows = cur.fetchall()

        out: list[dict] = []
        for r in rows:
            chunk_count = int(r["chunk_count"] or 0)
            uploaded_at: datetime | None = r["uploaded_at"]
            out.append(
                {
                    "filename": r["filename"],
                    "blob_url": r["blob_url"] or "",
                    "page_count": int(r["page_count"] or 0),
                    "chunk_count": chunk_count,
                    "uploaded_at": uploaded_at.isoformat() if uploaded_at else None,
                    "status": r["status"] or "pending",
                    "embedded": chunk_count > 0,
                }
            )
        return out
