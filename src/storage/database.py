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
from src.storage.conninfo import prefer_ipv4_conninfo
from src.storage.evaluations import EvalStore
from src.storage.messages import MessageStore
from src.storage.uploads import UploadStore

class DBManager:
    def __init__(self, conninfo: str) -> None:
        self._conninfo = conninfo
        self._uploads = UploadStore(conninfo)
        self._messages = MessageStore(conninfo)
        self._chunks = ChunkStore(conninfo)
        self._evaluations = EvalStore(conninfo)

    @classmethod
    def create(cls) -> "DBManager":
        url = settings.database_url.strip()
        if not url:
            raise ValueError("DATABASE_URL is not set.")
        return cls(prefer_ipv4_conninfo(url))

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

    def delete_chunks(self, filename: str) -> None:
        return self._chunks.delete_chunks(filename)

    def sample_chunks(self, sample_n: int = 0) -> list[dict]:
        return self._chunks.sample_chunks(sample_n)

    def search_chunks(
        self,
        query_vector: list[float],
        query_text: str = "",
        k: int = 5,
        filenames: list[str] | None = None,
        search_mode: str = "hybrid",
    ) -> list[dict]:
        return self._chunks.search_chunks(query_vector, query_text, k, filenames, search_mode)

    # --- evaluations ---

    def save_eval_run(
        self,
        config: dict,
        retrieval_results: dict,
        answer_quality_results: dict | None = None,
    ) -> int:
        return self._evaluations.save_run(config, retrieval_results, answer_quality_results)

    def list_eval_runs(self, limit: int = 20) -> list[dict]:
        return self._evaluations.list_runs(limit)

    def get_latest_eval_run(self) -> dict | None:
        return self._evaluations.get_latest_run()

    def get_eval_run(self, run_id: int) -> dict | None:
        return self._evaluations.get_run(run_id)
