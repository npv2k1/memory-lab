import math


def recall_at_k(relevant_flags: list[bool], total_relevant: int, k: int) -> float:
    if total_relevant <= 0 or k <= 0:
        return 0.0
    hits = sum(1 for flag in relevant_flags[:k] if flag)
    return round(min(1.0, hits / total_relevant), 6)


def mean_reciprocal_rank(relevant_flags: list[bool], k: int) -> float:
    for rank, is_relevant in enumerate(relevant_flags[:k], start=1):
        if is_relevant:
            return round(1.0 / rank, 6)
    return 0.0


def ndcg_at_k(relevant_flags: list[bool], total_relevant: int, k: int) -> float:
    if total_relevant <= 0 or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, is_relevant in enumerate(relevant_flags[:k], start=1)
        if is_relevant
    )
    ideal_hits = min(total_relevant, k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return round(dcg / idcg, 6)
