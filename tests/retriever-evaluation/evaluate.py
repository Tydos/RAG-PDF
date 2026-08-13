#!/usr/bin/env python3
"""
Evaluate retrieval quality against a QA file or gold set.

Supported input formats:
  1. Gold set (from generate_gold_set.py):
       [{"question", "ground_truth", "ground_truth_contexts", "source_pdf", "page"}, ...]
  2. Keyword format:
       {"questions": [{"question", "expected_keywords", "source_pdf"}, ...]}
  3. Simple list:
       [{"question", "answer"}, ...]

Usage:
    python evaluate.py --qa tests/retriever-evaluation/qa_pairs.json
    python evaluate.py --qa tests/evaluation/gold_set.json --top-k 5 --out results.json
"""

import argparse
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.evaluation.retrieval_metrics import aggregate_scores, score_retrieval
from src.evaluation.runner import load_qa_file

MODES = ["hybrid", "semantic", "keyword"]


def evaluate(api, qa, is_gold, mode, top_k, rerank):
    precision_scores, recall_scores = [], []
    for item in qa:
        try:
            resp = requests.post(
                f"{api}/chat",
                json={
                    "question": item["question"],
                    "top_k": top_k,
                    "search_mode": mode,
                    "persist": False,
                    "rerank": rerank,
                },
                timeout=60,
            ).json()
            chunks = resp.get("chunks") or []
        except Exception:
            chunks = []

        precision, recall, _, _ = score_retrieval(chunks, item["references"], is_gold)
        precision_scores.append(precision)
        recall_scores.append(recall)

    return precision_scores, recall_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", required=True, help="Path to QA file (gold set or keyword format)")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    parser.add_argument("--rerank", action="store_true", help="Also evaluate with reranking enabled")
    args = parser.parse_args()

    qa, is_gold = load_qa_file(args.qa)
    n = len(qa)
    fmt = "gold set" if is_gold else "keyword"
    print(f"Loaded {n} questions ({fmt} format)")

    try:
        h = requests.get(f"{args.api}/health", timeout=10)
        body = h.json()
        if body.get("services", {}).get("database") is False:
            print("Warning: database unhealthy — results may be empty")
    except requests.exceptions.ConnectionError as e:
        print(f"API unreachable: {e}")
        return 1

    results = {}
    variants = [(mode, False) for mode in MODES]
    if args.rerank:
        variants.extend((mode, True) for mode in MODES)

    print(f"\n{'MODE':<16} {'Precision@k':>12} {'Recall@k':>10} {'F1':>8}")
    print("-" * 50)
    for mode, rerank in variants:
        p_scores, r_scores = evaluate(args.api, qa, is_gold, mode, args.top_k, rerank)
        summary = aggregate_scores(p_scores, r_scores)
        p, r, f1 = summary["precision"], summary["recall"], summary["f1"]
        hits = int(r * n)
        label = f"{mode}+rerank" if rerank else mode
        print(f"{label:<16} {p:>11.1%}  {r:>8.1%} ({hits}/{n})  {f1:>6.1%}")
        results[label] = {
            **summary,
            "mode": mode,
            "rerank": rerank,
            "per_question": [
                {
                    "id": i + 1,
                    "question": qa[i]["question"],
                    "precision": p_scores[i],
                    "recall": r_scores[i],
                }
                for i in range(n)
            ],
        }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
