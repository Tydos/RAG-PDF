# Agent guide (required for all `src/` changes)

When writing or changing Python in this repo, prioritize **small files**, **KISS**, and **DRY**. Reuse before you write. Document rationale in [`CHANGELOG.md`](CHANGELOG.md), not in code comments. **Always explain what you did verbosely** in your session reply. **Write code assuming concurrency and authentication** — even when the app does not enforce auth yet.

## Session replies — explain everything verbosely

After every task (code, config, docs, review, or refactor), give a **detailed** explanation to the user. Do not reply with only “done” or a one-line summary.

Include:

- **What changed** — files, functions, routes, or config keys touched and how behavior differs now vs before.
- **Why** — problem being solved, alternatives considered, and why you chose this approach.
- **Reuse** — what existing code you searched for, found, reused, or extended (see **Before writing new code**).
- **Edge cases and risks** — see **Edge cases, errors, and optimizations**; always cover this for changed behavior.
- **What you did not do** — deferred items, out-of-scope work, or fixes you skipped and why.
- **How to verify** — commands run (`ruff`, `pytest`) and how the user can confirm the change works.

Write in clear prose (headings and bullets are fine). The user should understand the full story without opening the diff.

## Code review and improvement requests

When the user asks you to **review**, **improve**, or **audit** code (a file, module, or PR):

1. **Read [`AGENT.md`](AGENT.md) first** — treat it as the checklist for your reply and any suggested changes. Apply every section that fits (KISS/DRY, concurrency, auth, dependencies, no monoliths, etc.).
2. **Read the target file(s) in full** — and enough related code (imports, callers, tests, stores/services) to understand real behavior, not just the snippet shown.
3. **Search for existing patterns** — before suggesting new helpers or libraries, check whether the repo already solves the same problem elsewhere.
4. **Reply verbosely** — do not give a thin “looks good” or bullet list of nits alone. Structure the review around:
   - **Overview** — what the code does and overall assessment.
   - **Strengths** — what is already aligned with `AGENT.md`.
   - **Issues** — grouped by severity (blocking / medium / low), with file references.
   - **Edge cases, errors, optimizations** — full pass per **Edge cases, errors, and optimizations** (OOM, N+1, races, auth gaps, timeouts, etc.).
   - **Concrete improvements** — specific, actionable refactors (split file, reuse X, add test Y); say what you would **not** change and why.
5. **Do not edit code unless asked** — review-only requests get a written review first. If they ask you to implement fixes, then change code and follow the rest of this guide (including `CHANGELOG.md` when you ship changes).

## Before writing new code

**Always search for existing logic first.** Before adding a function, route, query, or helper:

1. Grep or read `src/` (and `tests/`) for the same or similar behavior — same endpoint shape, SQL, HTTP call, validation, error handling, etc.
2. Prefer extending an existing module, store, service, or protocol over creating a parallel implementation.
3. If something almost fits, refactor the existing code minimally so both call sites share it — do not copy-paste and tweak.
4. Tell the user what you found (file + symbol) and whether you reused, extended, or had to add new code — and why.

## Dependencies — no new libraries without approval

**Do not add new Python packages** unless the user explicitly approves.

- No new entries in `pyproject.toml`, `requirements*.txt`, or `uv.lock` without asking first.
- Solve problems with the **stdlib** and **dependencies already in the project** (check `pyproject.toml` / `uv.lock`).
- If a new library would help, **stop and ask** — name the package, why it is needed, and what you considered from the existing stack.
- Bumping versions of existing deps is fine when required for a fix; adding net-new packages is not.

## Design: small files, KISS, DRY — no monoliths

**No monoliths.** Do not add large all-in-one modules (e.g. a single `main.py` or `utils.py` that owns routes, SQL, HTTP, and business rules). Split by domain and responsibility.

**Default to small, focused modules.** One file should do one job (e.g. routes for chat, retrieval metrics, a single store). If a file is growing past ~200 lines or mixing unrelated concerns, split it before adding more.

**KISS — keep it simple.**

- Choose the straightforward approach over clever abstractions.
- Avoid extra layers (base classes, generic helpers, config indirection) until a second real use case appears.
- Prefer inline logic in the route or service when it is short and local; extract only when reuse or clarity demands it.

**DRY — don't repeat yourself.**

- When the same logic appears twice, extract it to a shared function or module — but keep that extraction small and named for what it does.
- Do not create a catch-all `utils.py` or mega-module; put shared code next to the domain it serves (`src/evaluation/`, `src/ingestion/`, etc.).
- Reuse existing helpers and protocols (`src/interfaces.py`, stores, services) instead of copying SQL or HTTP calls.

**Balance KISS and DRY:** duplication in two places is a smell; a premature abstraction used once is worse. Extract on the second use, not the first.

## Configuration — single source in `src/config.py`

**All constants and config values belong in [`src/config.py`](src/config.py).** Do not define tunable limits, feature flags, URLs, model names, paths, or env-backed values in route handlers, services, or other modules.

- Add new settings as fields on the `Settings` class (with sensible defaults). Use env vars via pydantic-settings field names (e.g. `max_upload_bytes` → `MAX_UPLOAD_BYTES`).
- Import config everywhere else: `from src.config import settings`. For module-level path constants defined in `config.py` (e.g. `PROJECT_ROOT`), import those by name from the same module.
- **Do not** call `os.getenv()` outside `src/config.py`.
- **Do not** duplicate the same limit or default in multiple files — if two modules need it, it lives in `Settings` once.

**Exceptions** (keep local; do not move to config unless they become tunable or shared):

- SQL DDL and one-off implementation literals in a single private helper
- Docstring examples and test-only fixtures

When reviewing or refactoring, flag module-level `UPPER_CASE` assignments outside `src/config.py` and move them unless they clearly fit an exception above.

## Concurrency and authentication (design defaults)

**Assume multiple users and overlapping requests.** Do not write code as if only one request runs at a time.

**Concurrency**

- Shared mutable state on `app.state` or module globals must be safe under concurrent async requests (no unguarded read-modify-write, no request-scoped data stored on shared objects).
- Background tasks (`BackgroundTasks`, indexing, eval) can race with API calls on the same resource (re-upload, delete while indexing) — design for idempotency, status checks, and clear failure modes.
- Database writes should avoid lost updates where it matters (use transactions, `ON CONFLICT`, or explicit delete-then-insert patterns already in the codebase).
- Do not rely on in-memory caches or singletons for per-user or per-tenant data without a concurrency story.
- Long work belongs off the critical request path when possible; document blocking handlers and timeout risk in your reply.

**Authentication and authorization**

- Treat every mutating or data-exposing endpoint as **eventually authenticated** — structure code so auth can be added without rewriting business logic.
- Do not trust client-supplied identifiers for ownership (filename, user id, tenant id); plan for server-side identity from a future auth layer.
- Fail closed: missing or invalid credentials should deny access, not silently fall back to public or admin behavior.
- Scope data access by identity where relevant (uploads, chat history, eval runs) — avoid global “list everything” patterns that cannot be partitioned later.
- Never log secrets (`HF_TOKEN`, `CLAUDE_TOKEN`, storage keys, API keys) or echo them in error responses.
- When auth is not wired yet, **call out the gap** in your reply and in `CHANGELOG.md`; do not treat open endpoints as permanent.

## Edge cases, errors, and optimizations (required in your reply)

For **every function or behavior you add or materially change**, give the user a **verbose** breakdown in your session reply (not buried in code). Cover all that apply:

**Resource and scale**

- **OOM / memory** — full buffering (e.g. reading entire uploads into RAM), unbounded lists, loading whole gold sets or result sets at once.
- **N+1 queries** — loops that hit the DB or an HTTP API per row/chunk/message; prefer batching, joins, or `IN` queries.
- **Connection / pool exhaustion** — opening a new DB or HTTP client per request instead of reusing lifespan-scoped services.
- **Timeouts** — long sync work in request handlers (eval runs, large embed batches), missing timeouts on external calls.

**Correctness and failure modes**

- Empty or missing inputs, partial failures, race conditions (e.g. re-upload while indexing), idempotency.
- Concurrent requests touching the same upload, chunk set, or eval run; duplicate background jobs.
- Which errors are handled vs propagated; what the API returns (status code, degraded vs hard fail).
- Side effects on failure (e.g. DB row deleted but blob left in storage).

**Security, authentication, and data**

- Missing or future auth on destructive or data-bearing routes; IDOR via filename or run id.
- Unsanitized filenames, path traversal, leaking tokens in logs or responses.

**Optimizations you considered or applied**

- Batching, caching, indexes, background tasks, streaming, clamping limits — and tradeoffs (complexity vs gain).

If none apply, say so explicitly. If you defer a fix, say what you deferred and why.

## Comments and in-code documentation

- **Do not** add multiline or block comments describing temporary fixes, patches, workarounds, or “TODO: clean up later.”
- **Do not** narrate implementation history in code (`# changed because X`, `# hack for Y`).
- Code should stay self-explanatory; **reasoning, tradeoffs, and temporary decisions belong in [`CHANGELOG.md`](CHANGELOG.md)** and in your reply to the user.
- Docstrings describe **current** contract (args, returns, raises), not session history.

## Lint and format

Always run from the repo root before finishing:

```bash
ruff check src/ tests/
ruff format src/ tests/
```

Fix lint issues; do not leave unformatted code.

## Tests

Add or update pytest coverage near the code you touch (`tests/`). New behavior needs tests; bug fixes should include a regression test when practical.

```bash
pytest tests/ -v
```

Keep the suite green.

## PEP 8

Follow PEP 8 (naming, imports, line length via Ruff defaults — 100 chars, spacing). Prefer clear names over clever ones. See **Design: small files, KISS, DRY** above for module size and structure.

## Docstrings (Google style)

Use Google-style docstrings for modules, classes, and public functions/methods. Include `Args:`, `Returns:`, `Raises:`, and `Yields:` sections when they apply. Private helpers may use a short one-liner when the name is sufficient.

Example:

```python
def fetch_photographs(self, limit: int, offset: int) -> list[dict]:
    """Fetch a page of photograph rows from the database.

    Args:
        limit: Maximum number of rows to return (1–100 at the API layer).
        offset: Number of rows to skip.

    Returns:
        A list of photograph dicts with id, filename, url, category, width,
        and height.

    Raises:
        RuntimeError: If ``DATABASE_URL`` is missing or the pool cannot be
            created.
    """
```

## Changelog (required — all rationale lives here)

Maintain [`CHANGELOG.md`](CHANGELOG.md) for **every** session where you change code, config, or docs.

- Add a dated section (`## YYYY-MM-DD`) at the top; newest first.
- Group bullets under `### Added`, `### Changed`, `### Fixed`, or `### Removed` as appropriate.
- **Every meaningful change needs a why** — not just what file moved. Include:
  - Reuse vs new code (e.g. “extended `ChunkStore.search_chunks` instead of duplicating SQL”).
  - Tradeoffs and known limitations (e.g. “eval still runs inline; background job deferred”).
  - Temporary or partial fixes — document them here, not in code comments.
- One bullet per logical change; avoid raw file lists without context.
- Skip trivial edits (typos, formatting-only) unless they fix user-visible behavior.
