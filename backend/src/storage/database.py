"""
    Database manager that provides high-level methods for uploads, messages, and chunks.
    create() initializes the connection pool based on settings.
    ping() checks database connectivity.
    Uploads: add_upload, set_status, remove_upload, list_uploads.
    Messages: add_message, get_messages.
    Chunks: save_chunks, delete_chunks, search_chunks.
    The actual SQL queries and logic are implemented in the respective store classes.
"""

import logging

import psycopg

from src.config import settings
from src.storage.chunks import ChunkStore
from src.storage.messages import MessageStore
from src.storage.uploads import UploadStore

class DBManager:
    def __init__(self, conninfo: str) -> None:
        self._conninfo = conninfo
        self._uploads = UploadStore(conninfo)
        self._messages = MessageStore(conninfo)
        self._chunks = ChunkStore(conninfo)

    @classmethod
    def create(cls) -> "DBManager":
        url = settings.database_url.strip()
        if not url:
            raise ValueError("DATABASE_URL is not set.")
        return cls(url)

    def ping(self) -> bool:
        try:
            with psycopg.connect(self._conninfo) as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            logging.exception("Database ping failed")
            return False

    # --- uploads ---

    def add_upload(self, filename: str, blob_url: str) -> None:
        return self._uploads.add_upload(filename, blob_url)

    def set_status(self, filename: str, status: str, page_count: int = 0) -> None:
        return self._uploads.set_status(filename, status, page_count)

    def remove_upload(self, filename: str) -> None:
        return self._uploads.remove_upload(filename)

    def list_uploads(self) -> list[dict]:
        return self._uploads.list_uploads()

    # --- messages ---

    def add_message(self, role: str, content: str, chunks: list[dict] | None = None) -> None:
        return self._messages.add_message(role, content, chunks)

    def get_messages(self, limit: int = 50) -> list[dict]:
        return self._messages.get_messages(limit)

    # --- chunks ---

    def save_chunks(
        self,
        filename: str,
        pages: list[int],
        indexes: list[int],
        texts: list[str],
        vectors: list[list[float]],
    ) -> None:
        return self._chunks.save_chunks(filename, pages, indexes, texts, vectors)

    def save_advisory_chunks(
        self,
        filename: str,
        pages: list[int],
        indexes: list[int],
        texts: list[str],
        vectors: list[list[float]],
        advisory_ids: list[str],
    ) -> None:
        return self._chunks.save_advisory_chunks(filename, pages, indexes, texts, vectors, advisory_ids)

    def delete_chunks(self, filename: str) -> None:
        return self._chunks.delete_chunks(filename)

    def search_chunks(
        self,
        query_vector: list[float],
        query_text: str = "",
        k: int = 5,
        filenames: list[str] | None = None,
        search_mode: str = "hybrid",
        source_type: str | None = None,
    ) -> list[dict]:
        return self._chunks.search_chunks(query_vector, query_text, k, filenames, search_mode, source_type)

    # --- advisory packages ---

    def list_advisory_packages(self) -> list[dict]:
        return self._uploads.list_advisory_packages()
