"""
MessageStore manages the 'messages' table, which stores user and assistant messages along with their associated chunks.
add_message inserts a new message into the database, optionally with associated chunks.
get_messages retrieves messages from the database, returning them in chronological order with their content and associated chunks.

future work: update/delete messages, pagination, filtering by role/date, etc.
"""

import json

import psycopg
from psycopg.rows import dict_row

from src.storage.table_definitions import _CREATE_MESSAGES_TABLE


class MessageStore:
    def __init__(self, conninfo: str) -> None:
        self._conninfo = conninfo
        self._create_tables()

    def _create_tables(self) -> None:
        with psycopg.connect(self._conninfo) as conn:
            conn.execute(_CREATE_MESSAGES_TABLE)
            conn.commit()

    def add_message(self, role: str, content: str, chunks: list[dict] | None = None) -> None:
        with psycopg.connect(self._conninfo) as conn:
            conn.execute(
                "INSERT INTO messages (role, content, chunks) VALUES (%s, %s, %s)",
                (role, content, json.dumps(chunks) if chunks else None),
            )
            conn.commit()

    def get_messages(self, limit: int = 50) -> list[dict]:
        with psycopg.connect(self._conninfo) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT role, content, chunks, created_at FROM messages ORDER BY created_at ASC LIMIT %s",
                    (limit,),
                )
                rows = cur.fetchall()
        return [
            {
                "role": r["role"],
                "content": r["content"],
                "chunks": r["chunks"] or [],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in rows
        ]
