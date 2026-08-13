#!/usr/bin/env python3
"""
Generate a gold set of QA pairs from indexed PDF chunks for RAG evaluation.

Usage (from project root):
    python tests/retriever-evaluation/generate_gold_set.py --sample-n 20
    python tests/retriever-evaluation/generate_gold_set.py --sample-n 0 --out tests/retriever-evaluation/gold_set.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
import anthropic
import psycopg
from psycopg.rows import dict_row
from tqdm import tqdm

from src.config import settings

sys.path.insert(0, str(Path(__file__).parents[2]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parents[3] / ".env")


logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

_DEFAULT_MODEL = "claude-haiku-4-5"

_SYSTEM_PROMPT = (
    "You are a QA pair generator for RAG evaluation. "
    "Given a passage, write exactly one factual question answerable from the passage alone, "
    "then write a concise reference answer (1-3 sentences). "
    'Respond ONLY as valid JSON: {"question": "...", "answer": "..."} '
    "Do not include any other text, markdown, or explanation."
)


def fetch_chunks(sample_n: int) -> list[dict[str, object]]:
    with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
        if sample_n > 0:
            rows = conn.execute(
                "SELECT filename, page, chunk_index, content FROM chunks ORDER BY random() LIMIT %s",
                (sample_n,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT filename, page, chunk_index, content FROM chunks ORDER BY random()"
            ).fetchall()
    return list(rows)


def parse_qa(text: str) -> dict[str, str] | None:
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
    # Extract fields individually (handles malformed/double-object responses)
    q = re.search(r'"question"\s*:\s*"((?:[^"\\]|\\.)+)"', text)
    a = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)+)"', text)
    if q and a:
        return {"question": q.group(1), "answer": a.group(1)}
    return None


def generate_entry(
    client: anthropic.Anthropic,
    chunk: dict[str, object],
    model: str,
    max_tokens: int,
) -> dict[str, object] | None:
    try:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.2,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Passage:\n{chunk['content']}"}],
        )
        block = message.content[0] if message.content else None
        response_text = block.text if block and hasattr(block, "text") else ""
    except Exception as exc:
        logging.warning("Claude call failed (file=%s page=%s): %s", chunk["filename"], chunk["page"], exc)
        return None

    parsed = parse_qa(response_text)
    if not parsed:
        logging.warning("Could not parse JSON (file=%s page=%s): %r", chunk["filename"], chunk["page"], response_text[:120])
        return None

    answer = parsed.get("answer", "").strip()
    question = parsed.get("question", "").strip()
    if not answer or not question:
        logging.warning("Empty question or answer (file=%s page=%s)", chunk["filename"], chunk["page"])
        return None

    return {
        "question": question,
        "ground_truth": answer,
        "ground_truth_contexts": [chunk["content"]],
        "source_pdf": chunk["filename"],
        "page": chunk["page"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a gold set from indexed PDF chunks using Claude")
    parser.add_argument("--sample-n", type=int, default=50, help="Chunks to sample (0 = all)")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="Claude model ID")
    parser.add_argument("--out", default="tests/retriever-evaluation/gold_set.json", help="Output JSON path")
    parser.add_argument("--max-tokens", type=int, default=400, help="Max tokens per call")
    parser.add_argument("--delay", type=float, default=0.3, help="Seconds between API calls")
    args = parser.parse_args()

    if not settings.database_url:
        print("Error: DATABASE_URL not set in .env")
        return 1
    if not settings.claude_token:
        print("Error: CLAUDE_TOKEN not set in .env")
        return 1

    print(f"Model: {args.model}")
    print(f"Fetching chunks (sample_n={args.sample_n or 'all'})...")
    chunks = fetch_chunks(args.sample_n)
    if not chunks:
        print("No chunks found. Make sure PDFs are indexed.")
        return 1
    print(f"Sampled {len(chunks)} chunks.")

    client = anthropic.Anthropic(api_key=settings.claude_token)
    gold_set: list[dict[str, object]] = []
    skipped = 0

    for chunk in tqdm(chunks, desc="Generating QA pairs"):
        t0 = time.perf_counter()
        entry = generate_entry(client, chunk, args.model, args.max_tokens)
        elapsed = time.perf_counter() - t0
        status = "ok" if entry else "skipped"
        tqdm.write(f"  [{status}] file={chunk['filename']} page={chunk['page']}  time={elapsed:.2f}s")
        if entry:
            gold_set.append(entry)
        else:
            skipped += 1
        if args.delay > 0:
            time.sleep(args.delay)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict[str, object]] = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = []

    combined = existing + gold_set
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"\nDone: {len(gold_set)} new entries generated, {skipped} skipped.")
    print(f"Total entries in {out_path}: {len(combined)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
