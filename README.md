# Antares

Upload PDFs, ask questions. Hybrid search (semantic + keyword, RRF-fused) + LLM answers with inline citations and persistent chat history.

**Live → https://rag-pdf-fawn.vercel.app/**

![chat page](docs/chatpage.png)

![eval page](docs/evalpage.png)

## Stack

| | |
|---|---|
| Frontend | React 18 |
| Backend | FastAPI + Python 3.11 |
| Database | PostgreSQL + pgvector + tsvector (Supabase) |
| Embeddings | HuggingFace — `all-MiniLM-L6-v2` (384-dim) |
| LLM | HuggingFace — `meta-llama/Llama-3.2-1B-Instruct` or Claude |
| Storage | Supabase Storage |

## How it works

1. **Upload** — browser POSTs PDF to `/upload`; backend stores it in Supabase Storage
2. **Index** — background task: extract text → chunk (800 chars, 100 overlap) → embed → store in PostgreSQL
3. **Chat** — question → embed → hybrid search → LLM → answer with citations; full history saved per session

## Setup

```bash
# Backend
cd backend && uv sync
cp .env.example .env   # fill in values below
uvicorn src.main:app --reload

# Frontend
cd frontend && npm install && npm start
```

**.env**

```
DATABASE_URL=postgresql://postgres:[password]@db.[project].supabase.co:5432/postgres
SUPABASE_SERVICE_KEY=...
HF_TOKEN=hf_...
CLAUDE_TOKEN=sk-ant-...   # optional — used for evaluation
```

**frontend/.env**

```
REACT_APP_API_PREFIX=http://localhost:8000
REACT_APP_SUPABASE_URL=https://[project].supabase.co
REACT_APP_SUPABASE_ANON_KEY=...
REACT_APP_SUPABASE_BUCKET=files
```

> **Supabase setup**: disable RLS on the `uploads` and `chunks` tables, or grant service role full access.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness + DB status |
| POST | `/upload` | Upload PDF (multipart) — stores + queues indexing |
| GET | `/documents` | List documents with status and chunk count |
| DELETE | `/files/{filename}` | Delete document and all its chunks |
| POST | `/chat` | Chat with history (question, top_k, search_mode) |
| GET | `/history` | Full conversation history |
| POST | `/query` | Stateless search + LLM (no history) |
| GET | `/eval/summary` | Pre-computed retrieval + answer quality results |

## Database

![schema](docs/image.png)

- **`uploads`** — one row per PDF (`filename` PK, `blob_url`, `status`, `page_count`)
- **`chunks`** — text chunks with 384-dim vector + tsvector; cascades on delete
- **`messages`** — chat history (`role`, `content`, `chunks` JSONB)

## Evaluation

Results on a 20-question gold set from ML/AI textbooks (top-k=5):

| Mode | Precision@5 | Recall@5 | F1 |
|---|---|---|---|
| hybrid | 10% | 40% | 16% |
| semantic | 6% | 30% | 10% |
| keyword | 19% | 80% | 31% |

Keyword wins on this corpus because questions are generated directly from chunk text. Hybrid/semantic are expected to improve on paraphrased queries.

## Limitations

- No OCR — image-only PDFs are marked `skipped`
- No auth — chat history is global (single shared thread)
- Max 100 MB per PDF
- LLM context capped at last 6 turns
