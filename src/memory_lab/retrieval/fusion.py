from collections import defaultdict
from memory_lab.core.models import Evidence


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[tuple[str, float]]],
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[tuple[str, float, dict[str, float]]]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion (RRF)."""
    weights = weights or {}
    fused_scores: dict[str, float] = defaultdict(float)
    breakdowns: dict[str, dict[str, float]] = defaultdict(dict)

    for source_name, items in ranked_lists.items():
        weight = weights.get(source_name, 1.0)
        for rank, (item_id, raw_score) in enumerate(items, start=1):
            contribution = weight / (k + rank)
            fused_scores[item_id] += contribution
            breakdowns[item_id][source_name] = round(float(raw_score), 6)

    merged = [
        (item_id, round(score, 6), {**breakdowns[item_id], "rrf": round(score, 6)})
        for item_id, score in fused_scores.items()
    ]
    merged.sort(key=lambda row: (-row[1], row[0]))
    return merged


def fuse_evidence(
    dense: list[Evidence],
    sparse: list[Evidence],
    limit: int,
    *,
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[Evidence]:
    """Fuse dense and sparse document Evidence lists using RRF."""
    by_id: dict[str, Evidence] = {}
    for hit in dense:
        by_id[hit.chunk.id] = hit
    for hit in sparse:
        by_id.setdefault(hit.chunk.id, hit)

    fused = reciprocal_rank_fusion(
        {
            "dense": [(hit.chunk.id, hit.score) for hit in dense],
            "sparse": [(hit.chunk.id, hit.score) for hit in sparse],
        },
        k=k,
        weights={"dense": dense_weight, "sparse": sparse_weight},
    )

    results: list[Evidence] = []
    for chunk_id, score, breakdown in fused[:limit]:
        base = by_id[chunk_id]
        results.append(
            Evidence(
                chunk=base.chunk,
                score=score,
                source_url=base.source_url,
                score_breakdown=breakdown,
            )
        )
    return results
