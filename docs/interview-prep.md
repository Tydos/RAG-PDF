# Interview Prep — Antares (RAG-PDF)

Full preparation guide for discussing this project in AI/ML, backend, and full-stack interviews.

---

## Table of Contents

1. [Elevator Pitch](#1-elevator-pitch-30-seconds)
2. [System Design Deep Dive](#2-system-design-deep-dive)
3. [Walk Me Through the Code](#3-walk-me-through-the-code)
4. [Tradeoffs Deep Dive](#4-tradeoffs--deep-dive)
5. [Technical Q&A](#5-technical-qa)
6. [STAR Stories](#6-star-stories)
7. [System Design Model Answers](#7-system-design-model-answers)
8. [Known Issues & Improvements](#8-known-issues--improvements)
9. [Practice Script](#9-practice-walkthrough-script-5-minutes)
10. [Self-Quiz](#10-self-quiz)

---

## 1. Elevator Pitch (30 seconds)

> "I built **Antares**, a PDF question-answering app with a full RAG pipeline: upload PDFs, chunk and embed them into PostgreSQL with pgvector, retrieve with **hybrid search** (semantic + keyword fused via RRF), and generate grounded answers with inline citations. I instrumented the full path — embed, search, LLM latency — and built an eval harness measuring retrieval precision/recall and LLM-as-judge faithfulness. It's deployed on Vercel with a React frontend and FastAPI backend. The architecture is deliberately modular — Protocol-based interfaces let me swap embedders and LLMs without touching the orchestration layer."

---

## 2. System Design Deep Dive

### 2.1 Problem Statement

| Requirement | How the system addresses it |
|---|---|
| Users upload PDFs and ask questions | `/upload` + background indexing |
| Answers must be grounded | Retrieval-first prompt; "don't know" system instruction |
| Users need to verify claims | Inline `[filename p.N]` citations + source chips linking to PDF pages |
| Mixed query types (exact terms + paraphrases) | Hybrid search with RRF |
| Debuggability for portfolio/review | Per-stage latency, chunk inspector, eval dashboard |

### 2.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React 18)                              │
│  UploadSection ──POST /upload──▶ FastAPI                                │
│  DocumentsSection ──Supabase client──▶ uploads table (read)              │
│  ChatSection ──POST /chat──▶ FastAPI                                    │
│  App.js ──getHistory()──▶ messages table (read)                         │
│  EvalDashboard ──GET /eval/summary──▶ FastAPI                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      BACKEND (FastAPI + lifespan DI)                       │
│                                                                          │
│  IngestionService          _rag() orchestrator                           │
│    ├─ download PDF           ├─ embed query                              │
│    ├─ PDFParser              ├─ search_chunks (hybrid/semantic/keyword)  │
│    ├─ HuggingFace embed      └─ LLMResponseGenerator                     │
│    └─ ChunkStore.save                                                    │
└─────────────────────────────────────────────────────────────────────────┘
          │                    │                         │
          ▼                    ▼                         ▼
   Supabase Storage    PostgreSQL + pgvector    HF Router / Claude API
   (PDF blobs)         (chunks, uploads, msgs)   (embed + generate)
```

### 2.3 Design Decision: Split Data Plane

**What the code does:** Reads (`uploads`, `messages`) go through the **Supabase JS client** from the browser. Writes that need secrets (embedding, LLM, service-role storage upload) go through **FastAPI**.

| Benefit | Cost |
|---|---|
| Frontend lists documents without a backend proxy | Two sources of truth for "what the API exposes" |
| Chat history loads on app mount without extra API route | RLS misconfiguration = data exposure |
| Simpler React code — no `/documents` or `/history` routes needed | Harder to add auth/session logic later |
| Supabase anon key is designed for client reads | Business logic split across client + server |

**Interview line:**
> "I chose a **hybrid BFF pattern**: the backend owns the RAG compute path; the client owns low-risk CRUD reads. For production I'd centralize reads behind the API and add per-user row-level security."

### 2.4 Component Responsibilities

| Layer | Responsibility | Key files |
|---|---|---|
| **API / orchestration** | HTTP routes, DI, `_rag()` pipeline | `backend/src/main.py` |
| **Ingestion** | Download, parse, chunk, embed, persist | `ingestion/service.py`, `pdf_parser.py` |
| **Storage** | Schema bootstrap, hybrid SQL, chat history | `storage/chunks.py`, `uploads.py`, `messages.py` |
| **Inference** | Embed, prompt, generate | `embeddings.py`, `prompt_builder.py`, `rag_generator.py`, `llm_adapters.py` |
| **Contracts** | Swappable interfaces | `interfaces.py` |
| **Config** | All tunables in one place | `config.py` |
| **Eval** | Offline retrieval + answer quality | `tests/retriever-evaluation/` |

### 2.5 Request Lifecycles

#### Upload + Index (async, non-blocking)

```
User selects PDF
  → POST /upload (multipart)
  → upload_to_supabase() [service role, x-upsert]
  → db.add_upload(filename, url) [status=pending]
  → background_tasks.add_task(pipeline.index_document)
  → HTTP 200 immediately ("indexing in progress")

Background:
  → httpx GET blob_url → temp file
  → asyncio.to_thread(PDFParser.extract_chunks)  # CPU-bound, off event loop
  → embedder.embed(texts)  # parallel batches via asyncio.gather
  → delete_chunks + save_chunks
  → set_status(indexed | skipped | failed)
```

**Design intent:** User never waits for embedding. Failed indexing is recorded, not thrown to the client.

#### Chat (synchronous RAG)

```
POST /chat { question, top_k, search_mode, filenames? }
  → history = db.get_messages()          # ⚠ loads 50 oldest, not 6 latest
  → embed question                     # HF API, ~100-500ms
  → search_chunks()                      # pgvector + tsvector + RRF
  → generator.generate()               # Claude or Llama 1B
  → db.add_message(user) + db.add_message(assistant, chunks)
  → return { answer, chunks, latency }
```

#### Stateless query (eval / dev)

Same as chat but `history=[]` and does **not** persist messages. Used by `evaluate.py` for retrieval benchmarking.

### 2.6 Data Model

```
uploads (filename PK)
  ├── blob_url, status, page_count, uploaded_at
  └── 1:N → chunks (CASCADE DELETE)
              ├── page, chunk_index, content
              ├── embedding vector(384)
              └── content_tsv tsvector (GENERATED STORED)

messages
  ├── role (user | assistant)
  ├── content
  ├── chunks JSONB  ← full retrieval payload for citation UI
  └── created_at
```

**Why store chunks on assistant messages:**
- Frontend can render source chips without re-querying
- Audit trail: "what evidence did the model see?"
- Eval can compare faithfulness against stored context

### 2.7 Dependency Injection Pattern

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = DBManager.create()
    app.state.embedder = HuggingFaceEmbeddingService()
    app.state.generator = create_rag_generator()  # Claude if token else HF
    app.state.extractor = PDFParser()
    app.state.pipeline = IngestionService(db, embedder, extractor)
```

**Why lifespan, not per-request instantiation:**
- Embedding HTTP client is reused
- DB connection string parsed once
- LLM adapter selected once at startup (Claude vs HF)

---

## 3. Walk Me Through the Code

File-by-file narrative for "open the repo and explain it" interviews.

### 3.1 Entry Point — `backend/src/main.py`

**What it does:** Wires the app, defines HTTP routes, owns the single RAG orchestrator `_rag()`.

**Key sections:**

1. **Lifespan** — creates `DBManager`, `HuggingFaceEmbeddingService`, `create_rag_generator()`, `PDFParser`, `IngestionService`.

2. **Dependency aliases** — `DB`, `Embedder`, `Generator`, `Pipeline` as `Annotated[..., Depends(...)]`.

3. **`_rag()`** — the heart of the system:

```python
async def _rag(req, db, embedder, generator, history) -> tuple:
    question = req.question.strip()
    top_k = max(1, min(req.top_k, 20))

    with tracker.measure("embed"):
        query_vector = (await embedder.embed([question]))[0]

    with tracker.measure("search"):
        chunks = db.search_chunks(
            query_vector, query_text=question, k=top_k,
            filenames=req.filenames or None, search_mode=req.search_mode,
        )

    with tracker.measure("llm"):
        answer = await generator.generate(question, chunks, history)
```

**Talk through:**
- `top_k` clamped to `[1, 20]` — prevents abuse and runaway context
- `LatencyTracker` wraps each stage independently
- LLM failure is caught inside `generate()`; `/query` still returns chunks

4. **Routes:**
   - `POST /upload` — storage + DB row + background index
   - `POST /ingest` — same pipeline but URL already exists
   - `POST /query` — stateless RAG
   - `POST /chat` — RAG + persist messages
   - `GET /eval/summary` — pre-computed eval JSON

**Notable:** No `/documents`, `/history`, `/files/{filename}` — frontend uses Supabase directly.

### 3.2 Configuration — `backend/src/config.py`

| Setting | Default | What it controls |
|---|---|---|
| `pdf_chunk_size` | 800 | Character window |
| `pdf_chunk_overlap` | 100 | Sliding step = 700 chars |
| `hf_embed_model` | all-MiniLM-L6-v2 | 384-dim embeddings |
| `hf_llm_model` | Llama-3.2-1B-Instruct | Default generator |
| `claude_token` | optional | Switches generator to Claude |
| `llm_temperature` | 0.2 | Low creativity for faithfulness |

### 3.3 Contracts — `backend/src/interfaces.py`

Three `Protocol` classes define the seams: `DatabaseProtocol`, `EmbedderProtocol`, `GeneratorProtocol`, `ExtractorProtocol`.

**Why Protocols not ABCs:** Structural subtyping — `DBManager` never needs to inherit; it just implements the methods. Tests inject mocks without patching globals.

### 3.4 Ingestion — `backend/src/ingestion/`

#### `supabase_storage.py`
- Raw REST upload to Supabase Storage with **service role key**
- `x-upsert: true` — re-uploading same filename overwrites
- Returns public URL used for download during indexing

#### `service.py` — `IngestionService.index_document()`

```
1. Download PDF to temp file (httpx async)
2. extract_chunks in thread pool (pypdf is sync/blocking)
3. embed all chunk texts (parallel batches)
4. delete_chunks (idempotent re-index) + save_chunks
5. set_status(indexed | skipped | failed)
```

**Code details:**
- `asyncio.to_thread()` for PDF parsing — keeps event loop free
- `delete_chunks` before `save_chunks` — safe re-index on re-upload
- Empty `texts` → `status=skipped` (image-only PDFs)
- `finally` block deletes temp file

#### `pdf_parser.py` — `PDFParser`

```python
def _split_into_chunks(self, text):
    step = max(1, chunk_size - overlap)  # 700
    while i < len(text):
        piece = text[i : i + chunk_size].strip()
        i += step
```

- Iterates **per page** — page number stored as metadata
- `chunk_index` is per-page sequence (0, 1, 2… within each page)
- No cleaning of PDF artifacts (headers, hyphenation, columns)

### 3.5 Storage — `backend/src/storage/`

#### `database.py` — `DBManager`
Thin facade delegating to `UploadStore`, `MessageStore`, `ChunkStore`.

#### `uploads.py` — `UploadStore`
- `ON CONFLICT (filename) DO UPDATE` — re-upload resets to `pending`
- `list_uploads()` JOINs chunk count

#### `messages.py` — `MessageStore`

```sql
SELECT role, content, chunks, created_at
FROM messages ORDER BY created_at ASC LIMIT %s
```

**Bug to know cold:** Returns the **first** N messages chronologically. For conversations >50 turns, `PromptBuilder`'s `history[-6:]` uses old context, not recent. Fix: `ORDER BY DESC LIMIT 6` then reverse.

#### `chunks.py` — `ChunkStore` (most important for retrieval interviews)

**Schema bootstrap on init:**
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE chunks (... embedding vector(384) ...);
ALTER TABLE chunks ADD COLUMN content_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
CREATE INDEX chunks_embedding_idx ON chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv_idx ON chunks USING GIN (content_tsv);
```

**`search_chunks()` routing:**
```python
if search_mode == "semantic" or not query_text.strip():
    return self._search_semantic(...)
if search_mode == "keyword":
    return self._search_keyword(...)
return self._search_hybrid(...)
```

**Semantic search:**
```sql
SELECT ..., 1 - (embedding <=> %s::vector) AS score
FROM chunks [WHERE filename = ANY(...)]
ORDER BY embedding <=> %s::vector LIMIT k
```

**Keyword search:**
```python
tsq = self._to_tsquery(query_text, op="|")  # OR between tokens
WHERE content_tsv @@ to_tsquery('english', tsq)
ORDER BY ts_rank(...) DESC
```

**Hybrid search — whiteboard this SQL:**

```sql
WITH
  vec AS (
    SELECT id, ..., ROW_NUMBER() OVER (ORDER BY embedding <=> vec) AS rn
    FROM chunks WHERE ... ORDER BY embedding <=> vec LIMIT pool  -- pool = k*4
  ),
  fts AS (
    SELECT id, ..., ROW_NUMBER() OVER (ORDER BY ts_rank DESC) AS rn
    FROM chunks, to_tsquery(...) query
    WHERE content_tsv @@ query ... LIMIT pool
  ),
  fused AS (
    SELECT COALESCE(v.id, f.id), ...,
           COALESCE(1.0/(60+v.rn), 0) + COALESCE(1.0/(60+f.rn), 0) AS score
    FROM vec v FULL OUTER JOIN fts f ON f.id = v.id
  )
SELECT * FROM fused ORDER BY score DESC LIMIT k
```

**Walk through each clause:**
- `vec` CTE: top `4k` by cosine distance, rank 1 = best semantic match
- `fts` CTE: top `4k` by `ts_rank`, rank 1 = best keyword match
- `FULL OUTER JOIN`: chunk in semantic only OR keyword only still gets a score
- RRF: rank 1 semantic + rank 1 keyword → `1/61 + 1/61 ≈ 0.033`
- Hybrid uses `_to_tsquery(op="&")` — **stricter** than keyword-only (`op="|"`)

**Performance footgun:** `SELECT COUNT(*) FROM chunks` runs on every search for logging.

### 3.6 Inference — `backend/src/inference/`

#### `embeddings.py` — `HuggingFaceEmbeddingService`
- HF Router feature-extraction endpoint
- Batches via `asyncio.gather` — ingestion batches run in parallel
- Normalizes HF quirk: single input sometimes returns flat list
- Requires `HF_TOKEN`

#### `prompt_builder.py` — `PromptBuilder`

```python
SYSTEM_PROMPT = "Answer strictly from context... Cite as [filename p.N]..."

def build_messages(question, chunks, history):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history[-6:],   # last 6 turns only
        {"role": "user", "content": f"Context:\n{formatted_chunks}\n\nQuestion: {question}"},
    ]
```

- Context in **final user turn** — works with Claude and HF adapters
- No token counting — all `top_k` chunks included regardless of size

#### `llm_adapters.py`
- **`ClaudeAdapter`:** extracts `system` → Anthropic `system` field
- **`HuggingFaceAdapter`:** OpenAI-compatible chat completions

#### `rag_generator.py` — `LLMResponseGenerator`

```python
if not chunks:
    return "I couldn't find anything relevant..."
# else build messages → llm.generate()
# on exception or empty → same fallback string
```

**`create_rag_generator()`:** Claude if `claude_token` set, else HuggingFace Llama 1B.

### 3.7 Frontend — `frontend/src/`

#### `api/api.js`

| Function | Path | Why |
|---|---|---|
| `uploadPDF` | `POST /upload` | Needs service role for storage |
| `chat`, `query` | `POST /chat`, `/query` | Needs HF/Claude secrets |
| `listDocuments` | Supabase `uploads` select | Simple read, no secrets |
| `getHistory` | Supabase `messages` select | Loads on app mount |
| `deleteDocument` | Supabase delete + storage remove | Client-side cascade |

#### `App.js`
- On mount: `getHistory()` → populates chat thread
- Two views: `chat` (sidebar + chat) and `eval` (EvalDashboard)

#### `ChatSection.js`
- Optimistic UI: appends user message before API returns
- Source chips: `blob_url#page=N` deep links to PDF page
- Passes `latency` for DevPanel

#### `DevPanel.js`
- Latency bars normalized to `total`
- Chunk accordion: filename, page, score, raw content

### 3.8 Evaluation — `backend/tests/retriever-evaluation/`

#### `generate_gold_set.py`
1. `SELECT ... FROM chunks ORDER BY random() LIMIT N`
2. Claude generates `{question, answer}` per chunk
3. Output: gold set with `ground_truth_contexts`

#### `evaluate.py`
- POSTs `/query` per question per mode
- Relevance: 50% token overlap between chunk and ground truth
- Reports P@k, R@k, F1

#### `answer_quality.py`
- Claude judge: faithfulness + relevance (0–1)
- Results → `/eval/summary`

**Interview narrative:** "I separated retrieval eval from generation eval because you can have good retrieval with bad answers (weak LLM) or vice versa."

### 3.9 Tests — `backend/tests/test_inference.py`

**Covered:** embedding batching, prompt structure, history cap, both adapters, generator fallbacks.

**Not covered:** hybrid SQL integration, end-to-end RAG (smoke test only).

---

## 4. Tradeoffs — Deep Dive

### 4.1 Chunking Strategy

| Option | Chosen | Alternative | Tradeoff |
|---|---|---|---|
| Fixed char window | ✅ 800/100 overlap | Sentence-aware | Simple code; hurts recall on technical PDFs |
| Per-page chunks | ✅ | Cross-page merge | Page citations accurate; splits spanning concepts |
| Token-based sizing | ❌ | tiktoken budget | Char ≠ tokens; LLM context unpredictable |
| Semantic chunking | ❌ | Embed sentences, split at discontinuities | Best quality; expensive at ingest |
| Parent-child chunks | ❌ | Small retrieve, large generate | Industry standard; adds schema complexity |

### 4.2 Embedding Model

| Option | Chosen | Tradeoff |
|---|---|---|
| `all-MiniLM-L6-v2` (384d) | ✅ | Fast; weaker on domain terms |
| Remote HF API | ✅ | No GPU ops; 100–500ms latency; cold starts |
| Local sentence-transformers | ❌ | +100MB RAM; 5–20ms embed |
| Asymmetric E5 prefixes | ❌ | Better retrieval; needs query/passage prefixes |

**Critical detail:** Same model embeds chunks and queries. No `query:` / `passage:` prefix distinction.

### 4.3 Vector Index

| Option | Chosen | Tradeoff |
|---|---|---|
| IVFFlat | ✅ | Fast build; needs `lists`/`probes` tuning |
| HNSW | ❌ | Better recall at scale; more memory |
| Separate vector DB | ❌ | Managed ANN; another service |

**IVFFlat in code:** default `lists=100`, `probes=1`, no rebuild after bulk insert.

### 4.4 Hybrid Fusion

| Option | Chosen | Tradeoff |
|---|---|---|
| RRF (rank-based) | ✅ | No score calibration; proven robust |
| Weighted score sum | ❌ | Needs normalization; fragile |
| Cross-encoder rerank | ❌ | +50–100ms; big precision win |

**RRF constant `60`:** standard default from literature. Pool size `k*4` balances recall vs SQL cost.

### 4.5 LLM Selection

| Option | Chosen | Tradeoff |
|---|---|---|
| Llama 3.2 1B (HF default) | ✅ | Free; weak reasoning; slow on shared infra |
| Claude Haiku (optional) | ✅ | Better quality + SLA; costs money |
| Streaming | ❌ | User waits for full completion; simpler code |

### 4.6 Persistence & History

| Option | Chosen | Tradeoff |
|---|---|---|
| Store chunks JSONB on assistant messages | ✅ | Auditable; larger rows |
| Global single chat thread | ✅ | Simple demo; no multi-user |
| Client reads history from Supabase | ✅ | Fast load; bypasses backend auth |

### 4.7 Ingestion Architecture

| Option | Chosen | Tradeoff |
|---|---|---|
| FastAPI BackgroundTasks | ✅ | Zero infra; shares worker with requests |
| Celery/RQ worker | ❌ | Isolates ingest; adds queue ops |
| Full re-index on re-upload | ✅ | Simpler than incremental |

### 4.8 Error Handling

| Behavior | Tradeoff |
|---|---|
| Empty retrieval → no LLM call | Prevents hallucination; correct |
| LLM fail → "couldn't find" message | Misleading UX |
| Index fail → `status=failed` in DB | Good; UI can show state |
| `/query` returns chunks if LLM fails | Good for retrieval debugging |

---

## 5. Technical Q&A

### "Walk me through what happens when a user asks a question."

1. Frontend POSTs to `/chat` with `question`, `top_k`, `search_mode`
2. Backend embeds question via HF (`all-MiniLM-L6-v2`)
3. `ChunkStore.search_chunks()` runs semantic, keyword, or hybrid SQL
4. `PromptBuilder` formats chunks + system prompt + last 6 history turns
5. `LLMResponseGenerator` calls Claude or Llama 1B
6. Response saved to `messages` with chunks JSONB
7. `LatencyTracker` returns `embed`, `search`, `llm`, `total` ms

### "How does hybrid search work? Why RRF?"

Two retrievers → rank each → fuse with `1/(60+rank_sem) + 1/(60+rank_kw)`. RRF avoids comparing incompatible score scales (cosine similarity vs `ts_rank`).

### "How do you prevent hallucinations?"

System prompt grounding, empty-retrieval guard, citations, low temperature, Claude-as-judge faithfulness eval. Missing: reranking, context budget, post-generation verification.

### "How do you evaluate RAG quality?"

**Retrieval:** P@5, R@5, F1 on 20-question gold set.

| Mode | Precision@5 | Recall@5 |
|---|---|---|
| hybrid | 10% | 40% |
| semantic | 6% | 30% |
| keyword | 19% | 80% |

**Generation:** Claude judge faithfulness + relevance.

### "What are the bottlenecks?"

| Stage | Impact | Mitigation |
|---|---|---|
| LLM generation | 40–80% latency | Streaming, Claude Haiku, trim context |
| Query embedding | 100–500ms | LRU cache, local model |
| Naive chunking | Poor recall | Sentence/token chunks |
| No reranking | Irrelevant context | Retrieve 20, rerank to 5 |
| History SQL bug | Wrong context | `DESC LIMIT 6` |

### "Why Protocol interfaces?"

DI via `Protocol` enables mocks in tests and swapping HF/local embedder or Claude/Llama without touching `_rag()`.

### "Why not LangChain/LlamaIndex?"

Full control over hybrid SQL, RRF fusion, and latency instrumentation. Pipeline is ~15 files — frameworks don't solve the specific fusion problem.

---

## 6. STAR Stories

### Story A: Hybrid search design
- **Situation:** Semantic search missed exact PDF terminology; keyword missed paraphrases.
- **Task:** Improve retrieval without new infrastructure.
- **Action:** pgvector + generated `tsvector`, RRF in raw SQL, three search modes for A/B.
- **Result:** Keyword wins on verbatim gold set; hybrid default for production paraphrases; eval dashboard proves measurability.

### Story B: Observability as a feature
- **Situation:** RAG feels like a black box.
- **Task:** Make retrieval transparent.
- **Action:** `LatencyTracker`, dev panel chunk inspector, citation deep links.
- **Result:** Debug retrieval vs generation independently.

### Story C: Graceful degradation
- **Situation:** External HF API can fail or cold-start.
- **Action:** Empty chunks → no LLM; LLM exception → fallback; `/query` still returns chunks.
- **Result:** System stays inspectable when generation is down.

---

## 7. System Design Model Answers

### "Design a PDF RAG system from scratch. What do you ask?"

1. PDF types — text-native or scanned?
2. Corpus size — 100 docs or 100K?
3. Latency SLA — 2s ok or sub-500ms?
4. Multi-tenant? Auth?
5. Answer style — extractive vs abstractive?
6. Citation granularity — page, paragraph, bounding box?

### "How would you scale to 10K PDFs / 5M chunks?"

1. Dedicated worker queue for ingestion
2. Rebuild IVFFlat/HNSW after bulk load; tune `lists`/`probes`
3. Metadata pre-filter + retrieve 20 → rerank 5
4. Partition `chunks` by collection
5. Local/managed embedding service with SLA
6. Query embedding cache + semantic response cache

### "How do you know if a change improved the system?"

Run `evaluate.py` + `answer_quality.py` before/after. Target Recall@5 ≥ 60%, Faithfulness ≥ 0.75. Watch `LatencyTracker` logs. Run `pytest backend/tests/`.

---

## 8. Known Issues & Improvements

| Issue | Location | Fix |
|---|---|---|
| History loads oldest 50, not latest 6 | `messages.py` | `ORDER BY DESC LIMIT 6` |
| LLM error = "no context" message | `rag_generator.py` | Distinct error strings |
| `COUNT(*)` every search | `chunks.py` | Remove or cache |
| Keyword AND in hybrid vs OR in keyword | `chunks.py` | Consistent strategy |
| No IVFFlat rebuild after index | `chunks.py` | `REINDEX` after bulk insert |
| README API routes ≠ `main.py` | docs drift | Align or document Supabase bypass |

**Prioritized roadmap:**
1. Fix chat history SQL
2. Sentence/token chunking
3. Cross-encoder rerank (retrieve 20 → rerank 5)
4. Context token budget in `PromptBuilder`
5. Local embeddings or query cache
6. SSE streaming
7. Claude Haiku for production
8. CI retrieval eval gate

---

## 9. Practice Walkthrough Script (5 minutes)

1. **Start at `main.py`** — lifespan DI, `_rag()` orchestrator
2. **Follow upload** — `supabase_storage` → `add_upload` → `IngestionService.index_document`
3. **Open `pdf_parser.py`** — chunking, explain limitation
4. **Open `chunks.py`** — schema, generated tsvector, hybrid SQL CTEs
5. **Open `prompt_builder.py`** — system prompt, history cap
6. **Open `rag_generator.py`** — empty-chunk guard, adapter selection
7. **Frontend `api.js`** — Supabase reads vs backend RAG
8. **Eval folder** — gold set, retrieval metrics, Claude judge
9. **Close with tradeoffs** — chunking is #1 quality lever; RRF is solid; 1B LLM is demo-grade

---

## 10. Self-Quiz

1. Explain RRF formula and why rank beats raw scores.
2. Why does keyword beat hybrid on your eval set?
3. What happens if `HF_TOKEN` is missing at query time vs ingest time?
4. How are citations formatted and stored for the UI?
5. What's in the system prompt and why last 6 history turns?
6. How would you add streaming without rewriting retrieval?
7. What's the difference between `/query` and `/chat`?
8. How does `content_tsv` stay in sync with `content`?
9. What does `FULL OUTER JOIN` do in the hybrid CTE?
10. Why is `asyncio.to_thread` used in ingestion?

---

## Closing Line

> "This project shows I can ship a **measurable** RAG system — not just a demo chatbot. I know exactly where quality breaks down (chunking, embedding model, no rerank) and I have a prioritized plan to fix it. The hybrid retrieval and eval harness are the parts I'm most proud of."

---

See also: [interview-prep-quick.md](./interview-prep-quick.md) for a 5-minute version.
