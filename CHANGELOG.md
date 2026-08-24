# Changelog

All notable changes made in agent sessions are recorded here. Newest entries first.

## 2026-08-24

### Changed

- `src/inference/embeddings.py` — extended existing `HuggingFaceEmbeddingService` (no new module): added an `HFError` exception hierarchy (`MissingHFToken`, `HFAuthError`, `HFTimeoutError`, `InvalidEmbeddingDimensions`) so embedding failures can be caught under one base type, similar in spirit to `LLMAuthError` in `llm_adapters.py`.
- `src/inference/embeddings.py` — `_fetch_embeddings` now retries transient Hugging Face Inference API failures (HTTP 429, 502, 503, 504) up to three times with exponential backoff (`2**attempt` seconds); 401/403 fail immediately as `HFAuthError` without retry; timeouts and connection errors retry then raise `HFTimeoutError` or `HFError`.
- `src/inference/embeddings.py` — missing `HF_TOKEN` is checked in `__init__` and raises `MissingHFToken` at service construction time (app lifespan), replacing the previous per-call `RuntimeError` in `embed()`; callers that instantiate the service without a token will fail at startup rather than on first embed.
- `src/inference/embeddings.py` — `embed()` validates batch size (`size < 1` → `ValueError`) and adds Google-style docstrings on the module, exception classes, service, and methods; removed redundant `pass` and narrating comments; applied Ruff formatting (PEP 8 spacing, line wraps).

### Added

- `InvalidEmbeddingDimensions` exception class reserved for future response validation against `settings.embed_dim`; not raised yet.
- `tests/test_inference.py` — updated embedding tests for `MissingHFToken` at construction time; added coverage for `HFAuthError`, exhausted 429/timeout retries, 429-then-success, invalid batch size, and non-retryable HTTP errors.

### Fixed

- `src/inference/embeddings.py` — success path returns inside the retry loop so a failed-then-success path no longer risks referencing an unassigned `data` variable after exhausted retries.

### Known limitations

- Retry attempt count and backoff remain hardcoded in `_fetch_embeddings` (not yet in `src/config.py`).
- Non-retryable HTTP errors (e.g. 400, 500) still propagate as `httpx.HTTPStatusError` from `raise_for_status()`, not wrapped in `HFError`.

## Prior history

1. Migrated from Minio to Vercel blob storage for deploying, added code for creating a secure upload token that the frontend can use for directly uploading to blob store.
2. ElasticSearch server coudn't be setup/run on a serverless setup and hence migrated to PostgreSQL ts-vector based searching (alternative for traditional keyword search - less accurate/fast compared to ES but deployment was a priority for me).
3. Generators experimented were Gemini, Claude and HuggingFace Llama3.2-7B. I decided to go with HF because Gemini Free tier is unreliable, Claude would rack up on costs long term.
4. Chat persistence was added using a new messages table for the LLM to have persistent context (last 6 messages).
5. Split up requirements into prod/dev requirements for a cleaner codebase. pyproject.toml was required for configuring test paths and build setups on vercel.
6. Added LLM assisted gold-generation set for evaluating the RAG system. I chose haiku over Llama for better gold generation. RAGAS addition would be the next steps.
7. PDF parser class was updated to only upload clean text chunks to DB for improving RAG system. Tiktoken offers better performance for splitting tokens.
8. Added LLM as a judge logic for evaluating whether am i getting the right answers, and checked metrics such as answer faithfulness.
9. Used Claude Impeccable skills to polish exisiting UI into more accessible and clean UI/UX. Added a evaluation page to reduce the clutter and made upload section drag-and-drop.
10. Read the documentation for Connection Pooling, it is not suited for a serverless instances which get turned off after use, hence removed psyopg connection pool setup.
11. Added cross-encoder reranking via HuggingFace Inference API (retrieve wider pool, rerank to top-k) and wired an in-app Evaluation dashboard with persisted eval runs, retrieval metrics, and optional Claude-as-judge answer quality scoring.

## 2026-08-22

### Changed

- Split monolithic `src/main.py` into domain routers (`src/routes/pages.py`, `chat.py`, `documents.py`, `eval.py`, `health.py`), shared deps (`src/deps.py`), template context helpers (`src/routes/context.py`), and RAG orchestration (`src/services/rag.py`); `main.py` now only wires app, lifespan, middleware, and `include_router`.
- `AGENT.md` — require all constants and config values in `src/config.py`; no `os.getenv()` outside config; import via `settings` (and shared path constants like `PROJECT_ROOT`).
- `src/config.py` — centralize upload limit, CORS origins, eval list limits, gold-set sample cap, and `PROJECT_ROOT`; add `cors_origin_list` property.
- `src/main.py` — remove local `BASE_DIR`, `MAX_UPLOAD_BYTES`, and `os.getenv("CORS_ORIGINS")`; use `settings` and `PROJECT_ROOT` instead.
- `AGENT.md` — code review requests must read `AGENT.md` and the target files first, then reply verbosely against the full checklist (no drive-by nits; no code edits unless asked).
- `AGENT.md` — require designing for concurrent requests and future authentication (shared state, races, fail-closed, scoped data, document auth gaps).
- `AGENT.md` — require verbose session replies explaining what changed, why, reuse, risks, deferrals, and verification.
- `AGENT.md` — require explicit user approval before adding any new Python dependency.
- `AGENT.md` — require searching existing code before new implementations; verbose edge-case/error/optimization notes in session replies; no monoliths; no patch/workaround comments in code; all rationale in `CHANGELOG.md`.
- `AGENT.md` — added design guidance to prioritize small files, KISS, and DRY when structuring `src/` code.

### Added

- `AGENT.md` — Python style guide for `src/` changes (Ruff, pytest, PEP 8, Google docstrings).
- `CHANGELOG.md` — agent-maintained log of session changes; prior history migrated from `docs/CHANGE.md`.

### Removed

- `docs/CHANGE.md` — content moved into `CHANGELOG.md` prior history section.
