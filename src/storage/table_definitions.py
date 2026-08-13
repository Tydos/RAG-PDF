from src.config import settings


_CREATE_MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    chunks JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""
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
_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS chunks_filename_idx ON chunks(filename);",
    "CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING ivfflat (embedding vector_cosine_ops);",
    "CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (content_tsv);",
]
_CREATE_UPLOADS_TABLE = """
CREATE TABLE IF NOT EXISTS uploads (
    filename TEXT PRIMARY KEY,
    blob_url TEXT NOT NULL DEFAULT '',
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status TEXT NOT NULL DEFAULT 'pending',
    page_count INTEGER NOT NULL DEFAULT 0
);
"""
_CREATE_EVAL_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS eval_runs (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    retrieval_results JSONB NOT NULL DEFAULT '{}'::jsonb,
    answer_quality_results JSONB
);
"""
