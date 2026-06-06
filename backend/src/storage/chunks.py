"""
    ChunkStore manages the 'chunks' table, which stores text chunks and their vector embeddings on PostgreSQL.
    to_pg_vector converts a list of floats to the PostgreSQL vector literal format.
    to_tsquery converts a query string into a tsquery format for full-text search.
    save_chunks inserts multiple chunks into the database.
    delete_chunks removes all chunks associated with a filename.
    search_chunks performs a search based on the specified mode (semantic, keyword, hybrid) and returns matching chunks with their scores.
"""

import logging
import re

import psycopg
from psycopg.rows import dict_row

from src.config import settings

_CREATE_VECTOR_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector;"

_CREATE_CHUNKS_TABLE = f"""
CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    filename TEXT NOT NULL REFERENCES uploads(filename) ON DELETE CASCADE,
    page INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector({settings.embed_dim}) NOT NULL
);
"""

_ADD_TSVECTOR_COLUMN = """
ALTER TABLE chunks
  ADD COLUMN IF NOT EXISTS content_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
"""

_ADD_ADVISORY_COLUMNS = """
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS source_type TEXT NOT NULL DEFAULT 'pdf';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS advisory_id TEXT;
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS chunks_filename_idx ON chunks(filename);",
    "CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING ivfflat (embedding vector_cosine_ops);",
    "CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (content_tsv);",
]


class ChunkStore:
    def __init__(self, conninfo: str) -> None:
        self._conninfo = conninfo
        self._create_tables()

    def _create_tables(self) -> None:
        with psycopg.connect(self._conninfo) as conn:
            try:
                conn.execute(_CREATE_VECTOR_EXTENSION)
                conn.execute(_CREATE_CHUNKS_TABLE)
                conn.execute(_ADD_TSVECTOR_COLUMN)
                for stmt in _ADD_ADVISORY_COLUMNS.strip().split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        conn.execute(stmt)
                for stmt in _CREATE_INDEXES:
                    conn.execute(stmt)
                conn.commit()
            except Exception:
                conn.rollback()
                logging.exception(
                    "pgvector extension not available; /query will fail until it is installed"
                )

    @staticmethod
    def _to_pg_vector(values: list[float]) -> str:
        return "[" + ",".join(f"{float(v):.7f}" for v in values) + "]"

    @staticmethod
    def _to_tsquery(text: str, op: str = "&") -> str:
        tokens = re.findall(r"[A-Za-z0-9]+", text)
        joined = f" {op} ".join(tokens)
        return joined if tokens else "placeholder"

    def save_chunks(
        self,
        filename: str,
        pages: list[int],
        indexes: list[int],
        texts: list[str],
        vectors: list[list[float]],
    ) -> None:
        if not texts:
            return
        rows = [
            (filename, page, idx, text, self._to_pg_vector(vec))
            for page, idx, text, vec in zip(pages, indexes, texts, vectors)
        ]
        with psycopg.connect(self._conninfo) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (filename, page, chunk_index, content, embedding) "
                    "VALUES (%s, %s, %s, %s, %s::vector)",
                    rows,
                )
            conn.commit()

    def save_advisory_chunks(
        self,
        filename: str,
        pages: list[int],
        indexes: list[int],
        texts: list[str],
        vectors: list[list[float]],
        advisory_ids: list[str],
    ) -> None:
        if not texts:
            return
        rows = [
            (filename, page, idx, text, self._to_pg_vector(vec), adv_id)
            for page, idx, text, vec, adv_id in zip(pages, indexes, texts, vectors, advisory_ids)
        ]
        with psycopg.connect(self._conninfo) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (filename, page, chunk_index, content, embedding, source_type, advisory_id) "
                    "VALUES (%s, %s, %s, %s, %s::vector, 'advisory', %s)",
                    rows,
                )
            conn.commit()

    def delete_chunks(self, filename: str) -> None:
        with psycopg.connect(self._conninfo) as conn:
            conn.execute("DELETE FROM chunks WHERE filename = %s", (filename,))
            conn.commit()

    def search_chunks(
        self,
        query_vector: list[float],
        query_text: str = "",
        k: int = 5,
        filenames: list[str] | None = None,
        search_mode: str = "hybrid",
        source_type: str | None = None,
    ) -> list[dict]:
        vec = self._to_pg_vector(query_vector)
        if search_mode == "semantic" or not query_text.strip():
            return self._search_semantic(vec, k, filenames, source_type)
        if search_mode == "keyword":
            return self._search_keyword(query_text, k, filenames, source_type)
        return self._search_hybrid(vec, query_text, k, filenames, source_type)

    def _search_semantic(self, vec: str, k: int, filenames: list[str] | None, source_type: str | None) -> list[dict]:
        sql = (
            "SELECT filename, page, chunk_index, content, "
            "1 - (embedding <=> %s::vector) AS score FROM chunks "
        )
        params: list = [vec]
        conditions = []
        if filenames:
            conditions.append("filename = ANY(%s)")
            params.append(filenames)
        if source_type:
            conditions.append("source_type = %s")
            params.append(source_type)
        if conditions:
            sql += "WHERE " + " AND ".join(conditions) + " "
        sql += "ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([vec, k])

        with psycopg.connect(self._conninfo) as conn:
            total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        logging.info("search(semantic): %d/%d chunks (filter=%s, k=%d)", len(rows), total, filenames, k)
        return [
            {"filename": r["filename"], "page": int(r["page"]), "chunk_index": int(r["chunk_index"]),
             "content": r["content"], "score": float(r["score"])}
            for r in rows
        ]

    def _search_keyword(self, query_text: str, k: int, filenames: list[str] | None, source_type: str | None) -> list[dict]:
        tsq = self._to_tsquery(query_text, op="|")
        sql = (
            "SELECT filename, page, chunk_index, content, "
            "ts_rank(content_tsv, to_tsquery('english', %s)) AS score FROM chunks "
        )
        params: list = [tsq]
        conditions = ["content_tsv @@ to_tsquery('english', %s)"]
        params.append(tsq)
        if filenames:
            conditions.append("filename = ANY(%s)")
            params.append(filenames)
        if source_type:
            conditions.append("source_type = %s")
            params.append(source_type)
        sql += "WHERE " + " AND ".join(conditions) + " "
        sql += "ORDER BY score DESC LIMIT %s"
        params.append(k)

        with psycopg.connect(self._conninfo) as conn:
            total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        logging.info("search(keyword): %d/%d chunks (filter=%s, k=%d, tsq=%r)", len(rows), total, filenames, k, tsq)
        return [
            {"filename": r["filename"], "page": int(r["page"]), "chunk_index": int(r["chunk_index"]),
             "content": r["content"], "score": float(r["score"])}
            for r in rows
        ]

    def _search_hybrid(self, vec: str, query_text: str, k: int, filenames: list[str] | None, source_type: str | None) -> list[dict]:
        tsq = self._to_tsquery(query_text)
        pool_size = k * 4

        conditions = []
        if filenames:
            conditions.append("filename = ANY(%(filenames)s)")
        if source_type:
            conditions.append("source_type = %(source_type)s")
        row_filter = " AND ".join(conditions) if conditions else "TRUE"

        sql = f"""
            WITH
              vec AS (
                SELECT id, filename, page, chunk_index, content,
                       ROW_NUMBER() OVER (ORDER BY embedding <=> %(vec)s::vector) AS rn
                FROM chunks WHERE {row_filter}
                ORDER BY embedding <=> %(vec)s::vector LIMIT %(pool)s
              ),
              fts AS (
                SELECT id, filename, page, chunk_index, content,
                       ROW_NUMBER() OVER (ORDER BY ts_rank(content_tsv, query) DESC) AS rn
                FROM chunks, to_tsquery('english', %(tsq)s) query
                WHERE {row_filter} AND content_tsv @@ query
                ORDER BY ts_rank(content_tsv, query) DESC LIMIT %(pool)s
              ),
              fused AS (
                SELECT
                  COALESCE(v.id, f.id)                  AS id,
                  COALESCE(v.filename, f.filename)       AS filename,
                  COALESCE(v.page, f.page)               AS page,
                  COALESCE(v.chunk_index, f.chunk_index) AS chunk_index,
                  COALESCE(v.content, f.content)         AS content,
                  COALESCE(1.0 / (60 + v.rn), 0) + COALESCE(1.0 / (60 + f.rn), 0) AS score
                FROM vec v FULL OUTER JOIN fts f ON f.id = v.id
              )
            SELECT * FROM fused ORDER BY score DESC LIMIT %(k)s
        """
        params = {"vec": vec, "tsq": tsq, "pool": pool_size, "k": k, "filenames": filenames, "source_type": source_type}

        with psycopg.connect(self._conninfo) as conn:
            total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        logging.info("search(hybrid): %d/%d chunks (filter=%s, k=%d, tsq=%r)", len(rows), total, filenames, k, tsq)
        return [
            {"filename": r["filename"], "page": int(r["page"]), "chunk_index": int(r["chunk_index"]),
             "content": r["content"], "score": float(r["score"])}
            for r in rows
        ]
