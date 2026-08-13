import re


def _normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _normalize(text)))


def _chunk_matches_reference(chunk_content: str, reference: str) -> bool:
    chunk = _normalize(chunk_content)
    ref = _normalize(reference)
    if not chunk or not ref:
        return False
    if chunk == ref:
        return True
    if len(ref) >= 40 and ref in chunk:
        return True
    if len(chunk) >= 40 and chunk in ref:
        return True
    chunk_tokens = _tokens(chunk_content)
    ref_tokens = _tokens(reference)
    if not chunk_tokens or not ref_tokens:
        return False
    overlap = len(chunk_tokens & ref_tokens) / len(ref_tokens)
    return overlap >= 0.6


def _chunk_is_relevant(chunk: dict, references: list[str], is_gold: bool) -> bool:
    content = chunk.get("content", "")
    if is_gold:
        return any(_chunk_matches_reference(content, ref) for ref in references)
    text = _normalize(content)
    return any(_normalize(keyword) in text for keyword in references)


def score_retrieval(
    chunks: list[dict],
    references: list[str],
    is_gold: bool,
) -> tuple[float, float, float, float]:
    """Return precision, recall, context_precision, context_recall."""
    if not references:
        return 0.0, 0.0, 0.0, 0.0

    if not chunks:
        return 0.0, 0.0, 0.0, 0.0

    relevant_flags = [_chunk_is_relevant(chunk, references, is_gold) for chunk in chunks]
    relevant_count = sum(1 for flag in relevant_flags if flag)
    precision = relevant_count / len(chunks)

    if is_gold:
        matched_refs = 0
        for ref in references:
            if any(_chunk_matches_reference(chunk.get("content", ""), ref) for chunk in chunks):
                matched_refs += 1
        recall = matched_refs / len(references)
    else:
        text = _normalize(" ".join(chunk.get("content", "") for chunk in chunks))
        matched_keywords = sum(1 for keyword in references if _normalize(keyword) in text)
        recall = matched_keywords / len(references)

    first_hit_rank = next((idx + 1 for idx, flag in enumerate(relevant_flags) if flag), None)
    context_precision = 1.0 / first_hit_rank if first_hit_rank else 0.0
    context_recall = 1.0 if any(relevant_flags) else 0.0

    return precision, recall, context_precision, context_recall


def aggregate_scores(
    precision_scores: list[float],
    recall_scores: list[float],
    context_precision_scores: list[float] | None = None,
    context_recall_scores: list[float] | None = None,
) -> dict[str, float]:
    if not precision_scores:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }

    precision = sum(precision_scores) / len(precision_scores)
    recall = sum(recall_scores) / len(recall_scores)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    ctx_precision = (
        sum(context_precision_scores) / len(context_precision_scores)
        if context_precision_scores
        else 0.0
    )
    ctx_recall = (
        sum(context_recall_scores) / len(context_recall_scores)
        if context_recall_scores
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "context_precision": ctx_precision,
        "context_recall": ctx_recall,
    }
