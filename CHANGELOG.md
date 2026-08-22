# Changelog

All notable changes made in agent sessions are recorded here. Newest entries first.

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
