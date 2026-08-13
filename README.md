# Antares

A measurable RAG retrieval evaluation platform. Upload PDFs, index them into PostgreSQL + pgvector, then benchmark hybrid search, reranking, and answer quality on your own corpus.

**Live → [https://rag-pdf-fawn.vercel.app/](https://rag-pdf-fawn.vercel.app/)**

## What it measures

- **Retrieval quality** — precision@k, recall@k, and F1 across `hybrid`, `semantic`, and `keyword` modes, with and without cross-encoder reranking
- **Answer quality** — faithfulness, relevance, and hallucination rate via HF Llama 3.3 70B judge (optional)
- **Latency** — per-stage timing for embed, search, rerank, and LLM

Use the **Chat** view for interactive queries with citations. Use the **Evaluation** dashboard to run gold-set benchmarks and compare retrieval strategies.

## Stack


|            |                                                      |
| ---------- | ---------------------------------------------------- |
| App        | FastAPI + Jinja2 + HTMX                              |
| Database   | PostgreSQL + pgvector + tsvector                     |
| Embeddings | HuggingFace — `all-MiniLM-L6-v2` (384-dim)           |
| Reranker   | HuggingFace — `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM        | HuggingFace Llama 3.2 1B or Claude                   |
| Judge      | HuggingFace — `meta-llama/Llama-3.3-70B-Instruct`    |
| Storage    | Supabase Storage or MinIO (S3-compatible)            |




## How it works

1. **Upload** — POST a PDF to `/upload`; the app stores it and indexes in the background
2. **Index** — extract text → chunk (800 chars, 100 overlap) → embed → store in PostgreSQL
3. **Query** — embed question → retrieve a candidate pool → optionally rerank → generate grounded answer with citations
4. **Evaluate** — generate a gold set, then run `/eval` to compare retrieval modes, reranking lift, and answer quality

### Config sweeps

The in-app evaluation dashboard sweeps search mode (`hybrid` / `semantic` / `keyword`) and reranking on/off. You can also pick the answer-generation LLM (`claude` / `hf`) when running answer-quality scoring.

Chunk-size and embedding-model sweeps require re-ingestion: update `PDF_CHUNK_SIZE`, `HF_EMBED_MODEL`, or related settings in `.env`, re-upload documents, then run a fresh evaluation so every run is compared against the same gold-set hash.



## Setup



### Local dev (Postgres in Docker)

```bash
docker compose up postgres -d
uv sync
cp .env.example .env   # DATABASE_URL already points at localhost Postgres
# add HF_TOKEN and storage keys for uploads + inference
uv run uvicorn src.main:app --reload
```

Open **[http://localhost:8000](http://localhost:8000)** for chat, **[http://localhost:8000/eval](http://localhost:8000/eval)** for the evaluation dashboard.

### Generate a gold set

The fastest way is the **Generate Gold Set** button on the `/eval` dashboard — pick a sample size and click; it samples indexed chunks and generates QA pairs with the HF judge model (`HF_JUDGE_MODEL`, defaults to `meta-llama/Llama-3.3-70B-Instruct`), appending to the existing gold set. Requires `HF_TOKEN`.

Alternatively, run it from the CLI (uses Claude instead):

```bash
python tests/retriever-evaluation/generate_gold_set.py --sample-n 20
```

Requires indexed PDFs in the database and `CLAUDE_TOKEN`.

### Run offline evaluation (CLI)

```bash
python tests/retriever-evaluation/evaluate.py --qa tests/retriever-evaluation/gold_set.json --rerank
python tests/retriever-evaluation/answer_quality.py --qa tests/retriever-evaluation/gold_set.json --mode all
```



## API


| Method | Path                | Description                                                                 |
| ------ | ------------------- | --------------------------------------------------------------------------- |
| GET    | `/`                 | Chat UI                                                                     |
| GET    | `/eval`             | Evaluation dashboard                                                        |
| GET    | `/eval/history`     | Evaluation run history                                                      |
| GET    | `/eval/{run_id}`    | Single evaluation run detail                                                |
| GET    | `/eval/summary`     | Latest evaluation run (JSON)                                                |
| POST   | `/eval/run`         | Run retrieval evaluation on gold set (`llm`, `include_answer_quality`)      |
| POST   | `/eval/gold-set/generate` | Generate gold-set QA pairs from indexed chunks via HF judge model      |
| GET    | `/health`           | Liveness + DB status                                                        |
| POST   | `/upload`           | Upload PDF (multipart)                                                      |
| GET    | `/documents`        | List documents with status                                                  |
| DELETE | `/files/{filename}` | Delete document and chunks                                                  |
| POST   | `/chat`             | RAG query (`persist: false` for stateless eval calls; `rerank: true/false`) |
| GET    | `/history`          | Conversation history                                                        |




## Database

- `uploads` — one row per PDF (`filename` PK, `blob_url`, `status`, `page_count`)
- `chunks` — text chunks with 384-dim vector + tsvector
- `messages` — chat history (`role`, `content`, `chunks` JSONB)
- `eval_runs` — persisted evaluation results (`config`, `retrieval_results`, `answer_quality_results`)



## Configuration

Key environment variables (see `.env.example`):

```
DATABASE_URL=postgresql://...
HF_TOKEN=hf_...
HF_JUDGE_MODEL=meta-llama/Llama-3.3-70B-Instruct
CLAUDE_TOKEN=sk-ant-...          # optional, used for gold-set generation and chat LLM
RERANK_ENABLED=true              # default on
HF_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```



## Limitations

- No OCR — image-only PDFs are marked `skipped`
- No auth — chat history and eval runs are global
- Max 100 MB per PDF
- LLM context capped at last 6 turns
- Reranker requires `HF_TOKEN`; fails open if unavailable

