# Interview Prep — Quick Reference (5 min)

Condensed cheat sheet. Full guide: [interview-prep.md](./interview-prep.md).

---

## Elevator Pitch

> "Antares is a PDF RAG app: upload → chunk → embed (MiniLM, pgvector) → hybrid search (semantic + keyword, RRF) → LLM answer with citations. I instrumented embed/search/LLM latency and built retrieval + faithfulness evals. Modular Protocol-based design on FastAPI + React + Supabase."

---

## Architecture (one diagram)

```
Upload → Supabase Storage → [background] parse/chunk/embed → PostgreSQL (pgvector + tsvector)
Question → embed → hybrid search (top-k) → prompt + history → Claude/Llama → answer + citations
```

**Split data plane:** Frontend reads `uploads`/`messages` from Supabase; RAG (`/chat`, `/upload`) goes through FastAPI.

---

## Code Walkthrough (file order)

| # | File | Say this |
|---|---|---|
| 1 | `main.py` | Lifespan DI; `_rag()` = embed → search → generate; `top_k` clamped 1–20 |
| 2 | `ingestion/service.py` | Async download → `to_thread` parse → parallel embed → save |
| 3 | `pdf_parser.py` | 800-char sliding window per page; **main quality bottleneck** |
| 4 | `storage/chunks.py` | pgvector + generated tsvector; hybrid = RRF CTEs, pool `k*4` |
| 5 | `prompt_builder.py` | System grounding prompt; context in user turn; history[-6:] |
| 6 | `rag_generator.py` | Empty chunks → no LLM; Claude if token else Llama 1B |
| 7 | `frontend/api/api.js` | Supabase reads vs backend RAG writes |
| 8 | `tests/retriever-evaluation/` | Gold set, P/R/F1, Claude faithfulness judge |

---

## Hybrid Search (must know)

```sql
-- Two CTEs: top 4k semantic (cosine) + top 4k keyword (ts_rank)
-- FULL OUTER JOIN on chunk id
-- score = 1/(60+rank_sem) + 1/(60+rank_kw)   ← RRF
-- ORDER BY score DESC LIMIT k
```

**Why RRF:** Semantic scores and `ts_rank` aren't comparable; ranks are.

**Why keyword wins eval:** Gold questions generated from chunk text → literal overlap favors FTS (R@5: 80% keyword vs 40% hybrid).

---

## Eval Numbers (cite these)

| Mode | P@5 | R@5 |
|---|---|---|
| hybrid | 10% | 40% |
| semantic | 6% | 30% |
| keyword | 19% | 80% |

Low precision = many irrelevant chunks in top-k. Low semantic recall = chunking + embedding limits.

---

## Top Tradeoffs

| Decision | Pro | Con |
|---|---|---|
| MiniLM 384d | Fast, cheap | Weak on technical PDFs |
| Char chunking (800/100) | Simple | Splits mid-sentence; hurts recall |
| PostgreSQL only | One DB for vectors + FTS + metadata | IVFFlat tuning needed at scale |
| RRF hybrid | Robust, no score calibration | Not tuned; no rerank stage |
| Llama 1B default | Free tier | Weak + slow on HF shared infra |
| BackgroundTasks ingest | Zero infra | Shares worker with API |
| Global chat thread | Simple demo | No auth/sessions |
| Supabase client reads | Fast UI | Split architecture, RLS risk |

---

## Known Bugs (shows depth)

1. **`messages.py`:** `ORDER BY ASC LIMIT 50` → oldest messages, not latest 6 for long chats
2. **`rag_generator.py`:** LLM failure message same as "no context found"
3. **`chunks.py`:** `COUNT(*)` on every search; hybrid uses AND, keyword mode uses OR

---

## "What would you do next?" (always have this)

1. Fix history SQL (`DESC LIMIT 6`)
2. Sentence/token chunking
3. Cross-encoder rerank (20 → 5)
4. Context token budget (~1500 tokens evidence)
5. Local embeddings or query cache
6. SSE streaming
7. Claude Haiku for prod
8. CI retrieval eval gate

---

## Curveball One-Liners

| Question | Answer |
|---|---|
| Production-ready? | Strong MVP/portfolio; needs auth, OCR, rerank, off cold-start APIs |
| Scale to 1M chunks? | Tune IVFFlat/HNSW, partition, worker queue, rerank, embedding SLA |
| Why not LangChain? | Wanted control over hybrid SQL + RRF + latency instrumentation |
| Hardest bug? | Chat history fetches wrong window after 50 messages |
| Prevent hallucinations? | Grounded prompt + empty guard + citations + faithfulness eval |

---

## Vocabulary

RRF · ANN/IVFFlat · FTS/tsvector · Faithfulness · Precision@k · Recall@k · Grounding · Cold start · Chunk overlap

---

## 5-Min Walkthrough Script

1. `main.py` → `_rag()`
2. Upload path → `IngestionService`
3. `chunks.py` → hybrid SQL
4. `prompt_builder.py` → grounding
5. `api.js` → split architecture
6. Eval results → honest metrics
7. Close: "chunking is #1 lever; hybrid RRF is solid; roadmap is clear"

---

## Closing Line

> "I shipped a measurable RAG system with hybrid retrieval and eval harnesses. I know where it breaks down and have a prioritized fix plan."
