from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from src.config import settings
from src.inference.llm_adapters import HuggingFaceAdapter
from src.interfaces import DatabaseProtocol

_SYSTEM_PROMPT = (
    "You are a QA pair generator for RAG evaluation. "
    "Given a passage, write exactly one factual question answerable from the passage alone, "
    "then write a concise reference answer (1-3 sentences). "
    'Respond ONLY as valid JSON: {"question": "...", "answer": "..."} '
    "Do not include any other text, markdown, or explanation."
)


def _gold_set_path() -> Path:
    return Path(settings.eval_gold_set_path)


def _parse_qa(text: str) -> dict[str, str] | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    q = re.search(r'"question"\s*:\s*"((?:[^"\\]|\\.)+)"', text)
    a = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)+)"', text)
    if q and a:
        return {"question": q.group(1), "answer": a.group(1)}
    return None


async def _generate_entry(
    adapter: HuggingFaceAdapter,
    chunk: dict,
    max_tokens: int,
) -> dict | None:
    try:
        text = await adapter.generate(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Passage:\n{chunk['content']}"},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
    except Exception as exc:
        logging.warning(
            "Gold-set generation call failed (file=%s page=%s): %s",
            chunk.get("filename"),
            chunk.get("page"),
            exc,
        )
        return None

    parsed = _parse_qa(text)
    if not parsed:
        logging.warning(
            "Could not parse gold-set JSON (file=%s page=%s): %r",
            chunk.get("filename"),
            chunk.get("page"),
            text[:120],
        )
        return None

    question = (parsed.get("question") or "").strip()
    answer = (parsed.get("answer") or "").strip()
    if not question or not answer:
        return None

    return {
        "question": question,
        "ground_truth": answer,
        "ground_truth_contexts": [chunk["content"]],
        "source_pdf": chunk.get("filename"),
        "page": chunk.get("page"),
    }


async def generate_gold_set(
    sample_n: int,
    db: DatabaseProtocol,
    out_path: Path | None = None,
) -> dict:
    """Generate a gold set of QA pairs from indexed chunks using the strong HF judge model.

    Appends to the existing gold set file (does not overwrite prior entries).
    """
    if not settings.hf_token:
        raise RuntimeError("HF_TOKEN is not configured.")

    target = out_path or _gold_set_path()
    chunks = db.sample_chunks(sample_n)
    if not chunks:
        return {"generated": 0, "skipped": 0, "total": 0, "path": str(target)}

    adapter = HuggingFaceAdapter(
        model=settings.hf_judge_model,
        token=settings.hf_token,
        timeout=90,
    )

    new_entries: list[dict] = []
    skipped = 0
    for chunk in chunks:
        entry = await _generate_entry(adapter, chunk, settings.hf_judge_max_tokens)
        if entry:
            new_entries.append(entry)
        else:
            skipped += 1

    target.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except (json.JSONDecodeError, OSError):
            existing = []

    combined = existing + new_entries
    target.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "generated": len(new_entries),
        "skipped": skipped,
        "total": len(combined),
        "path": str(target),
    }
