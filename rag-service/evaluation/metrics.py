"""Retrieval metrics from week 5 (evaluation/eval.py): keyword-based MRR, nDCG and coverage."""

import math


def reciprocal_rank(keyword: str, docs: list[str]) -> float:
    k = keyword.lower()
    for rank, doc in enumerate(docs, start=1):
        if k in doc.lower():
            return 1.0 / rank
    return 0.0


def _dcg(relevances: list[int], k: int) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg(keyword: str, docs: list[str], k: int = 10) -> float:
    kw = keyword.lower()
    relevances = [1 if kw in d.lower() else 0 for d in docs[:k]]
    ideal = _dcg(sorted(relevances, reverse=True), k)
    return _dcg(relevances, k) / ideal if ideal > 0 else 0.0


def retrieval_scores(keywords: list[str], docs: list[str], k: int = 10) -> dict:
    """Averages over keywords, like week 5. Returns None values when there are no keywords."""
    if not keywords:
        return {"mrr": None, "ndcg": None, "coverage": None}
    rr = [reciprocal_rank(kw, docs) for kw in keywords]
    return {
        "mrr": sum(rr) / len(rr),
        "ndcg": sum(ndcg(kw, docs, k) for kw in keywords) / len(keywords),
        "coverage": sum(1 for r in rr if r > 0) / len(keywords) * 100,
    }
