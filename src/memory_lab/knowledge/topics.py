"""c-TF-IDF Topic Map builder over Vietnamese-segmented phrases and entities.

Replaces opaque 2D embedding scatter plots for end users with an interpretable
Topic Map ("What is my knowledge base about?"): each cluster has a clean,
meaningful multi-word label, size proportional to chunk count, dominant source
document, top entities, and representative excerpts.
"""

from collections import Counter, defaultdict
import math
from typing import Any
import numpy as np
from memory_lab.documents.enrichment.vi_tokenizer import extract_topic_phrases


def _humanize_phrase(phrase: str) -> str:
    clean = " ".join(phrase.strip().split())
    if not clean:
        return "Chủ đề chung"
    # Preserve tech acronyms/proper case if already uppercase/mixed
    if clean.isupper() or any(ch.isupper() for ch in clean[1:]):
        return clean
    return clean[0].upper() + clean[1:]


def _dedupe_phrases(ranked: list[tuple[str, float]], limit: int = 5) -> list[str]:
    selected: list[str] = []
    for phrase, _score in ranked:
        low = phrase.lower()
        # Skip phrases strictly contained in an already selected longer phrase, or vice versa
        if any(low in s.lower() or s.lower() in low for s in selected):
            continue
        selected.append(_humanize_phrase(phrase))
        if len(selected) >= limit:
            break
    return selected


def _kmeans_cosine(matrix: np.ndarray, k: int, *, max_iter: int = 25) -> np.ndarray:
    n = matrix.shape[0]
    if k <= 1 or n <= 1:
        return np.zeros(n, dtype=int)
    k = min(k, n)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = matrix / norms

    # Deterministic farthest-first seeding
    centers = [unit[0].copy()]
    chosen = {0}
    for _ in range(1, k):
        sims = np.max(unit @ np.stack(centers).T, axis=1)
        for idx in chosen:
            sims[idx] = 2.0
        next_idx = int(np.argmin(sims))
        chosen.add(next_idx)
        centers.append(unit[next_idx].copy())
    centroids = np.stack(centers)

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        sims = unit @ centroids.T
        new_labels = np.argmax(sims, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cid in range(k):
            members = unit[labels == cid]
            if len(members) > 0:
                mean = members.mean(axis=0)
                norm = float(np.linalg.norm(mean))
                centroids[cid] = mean / norm if norm > 0 else mean
    return labels


def build_topic_map(
    items: list[dict[str, Any]],
    *,
    n_topics: int | None = None,
) -> dict[str, Any]:
    """Cluster knowledge items (document chunks + memories) into labeled topic clusters via c-TF-IDF.

    Each item dict expects:
      - id: str
      - source_type: 'document' | 'memory'
      - source_label: str (filename or 'Ký ức')
      - document_id: str | None
      - version: str | None
      - heading: str
      - text: str
      - entities: list[dict] ({id, name, type})
    """
    if not items:
        return {
            "total_items": 0,
            "topic_count": 0,
            "topics": [],
        }

    # 1. Extract Vietnamese-segmented topic phrases per item
    doc_phrases: list[list[str]] = []
    df: Counter[str] = Counter()
    for item in items:
        heading = item.get("heading") or ""
        text = item.get("text") or ""
        # Give heading phrases extra weight
        phrases = extract_topic_phrases(text)
        if heading:
            h_phrases = extract_topic_phrases(heading)
            phrases = h_phrases * 2 + phrases
        # Also include recognized multi-word or tech entities as candidate terms
        for ent in item.get("entities") or []:
            ename = str(ent.get("name") or "").strip().lower()
            if ename and (" " in ename or ent.get("type") in {"technology", "organization", "person"}):
                phrases.append(ename)
        doc_phrases.append(phrases)
        for unique_p in set(phrases):
            df[unique_p] += 1

    # Build bounded TF-IDF vocabulary (top 250 informative phrases)
    n_items = len(items)
    vocab_candidates = [
        (term, count)
        for term, count in df.items()
        if (count >= 2 or n_items < 8) and count <= max(2, int(n_items * 0.92))
    ]
    if not vocab_candidates:
        vocab_candidates = list(df.items())
    vocab_candidates.sort(key=lambda pair: (-pair[1], -len(pair[0].split()), pair[0]))
    vocab = [term for term, _ in vocab_candidates[:250]]
    vocab_idx = {term: idx for idx, term in enumerate(vocab)}

    # Choose number of clusters
    if n_topics is None:
        if n_items <= 3:
            k = 1
        elif n_items <= 8:
            k = min(3, n_items)
        else:
            k = max(2, min(8, int(round(math.sqrt(n_items / 2.2)))))
    else:
        k = max(1, min(int(n_topics), n_items, 12))

    if vocab and k > 1:
        mat = np.zeros((n_items, len(vocab)), dtype=np.float32)
        for row_i, phrases in enumerate(doc_phrases):
            tf = Counter(phrases)
            for term, cnt in tf.items():
                col = vocab_idx.get(term)
                if col is not None:
                    idf = math.log((1 + n_items) / (1 + df[term])) + 1.0
                    # Slightly boost multi-word phrases so clusters form around real topics
                    multi_boost = 1.35 if " " in term else 1.0
                    mat[row_i, col] = (1.0 + math.log(cnt)) * idf * multi_boost
        # Add mild source grouping signal if all rows are zero
        if float(np.linalg.norm(mat)) == 0.0:
            labels = np.array([i % k for i in range(n_items)], dtype=int)
        else:
            labels = _kmeans_cosine(mat, k)
    else:
        labels = np.zeros(n_items, dtype=int)

    # 2. Compute class-based TF-IDF (c-TF-IDF) per cluster for crisp topic labels
    clusters_by_id: dict[int, list[int]] = defaultdict(list)
    for idx, cid in enumerate(labels.tolist()):
        clusters_by_id[cid].append(idx)

    # Document frequency across clusters
    cluster_phrase_counts: dict[int, Counter[str]] = {}
    cluster_df: Counter[str] = Counter()
    for cid, member_indices in clusters_by_id.items():
        c_counter: Counter[str] = Counter()
        for idx in member_indices:
            c_counter.update(doc_phrases[idx])
        cluster_phrase_counts[cid] = c_counter
        for term in c_counter:
            cluster_df[term] += 1

    num_clusters = max(1, len(clusters_by_id))
    topics_out: list[dict[str, Any]] = []

    for cid, member_indices in sorted(clusters_by_id.items(), key=lambda pair: -len(pair[1])):
        c_counter = cluster_phrase_counts[cid]
        total_terms = max(1, sum(c_counter.values()))
        scored_terms: list[tuple[str, float]] = []
        for term, cnt in c_counter.items():
            tf_c = cnt / total_terms
            idf_c = math.log(1.0 + num_clusters / max(1, cluster_df[term]))
            word_count = len(term.split())
            phrase_bonus = 1.65 if word_count >= 2 else 1.0
            scored_terms.append((term, tf_c * idf_c * phrase_bonus))
        scored_terms.sort(key=lambda pair: (-pair[1], -len(pair[0].split()), pair[0]))

        keywords = _dedupe_phrases(scored_terms, limit=5)

        # Heading consensus check: if most items in cluster share a heading, prefer it as label
        heading_counts = Counter(
            items[idx]["heading"] for idx in member_indices if items[idx].get("heading")
        )
        top_heading = heading_counts.most_common(1)[0] if heading_counts else None
        if top_heading and top_heading[1] >= max(2, int(len(member_indices) * 0.5)):
            label = top_heading[0]
        elif len(keywords) >= 2:
            label = f"{keywords[0]} · {keywords[1]}"
        elif keywords:
            label = keywords[0]
        else:
            # Fallback to dominant source filename
            src_counts = Counter(items[idx]["source_label"] for idx in member_indices)
            label = src_counts.most_common(1)[0][0]

        # Sources breakdown & dominant source (for color coding in Treemap)
        source_counter: Counter[str] = Counter(items[idx]["source_label"] for idx in member_indices)
        primary_source = source_counter.most_common(1)[0][0]

        # Top 5 entities in this topic cluster
        ent_counts: Counter[str] = Counter()
        ent_meta: dict[str, dict[str, str]] = {}
        for idx in member_indices:
            for ent in items[idx].get("entities") or []:
                eid = ent.get("id") or str(ent.get("name", "")).lower()
                if not eid:
                    continue
                ent_counts[eid] += 1
                ent_meta.setdefault(eid, {"id": eid, "name": ent.get("name", eid), "type": ent.get("type", "concept")})
        top_entities = [
            {**ent_meta[eid], "count": cnt, "mentions": cnt}
            for eid, cnt in ent_counts.most_common(5)
        ]

        # Representative chunks/memories (longest informative previews)
        reps = sorted(
            (items[idx] for idx in member_indices),
            key=lambda it: (len(it.get("entities") or []), len(it.get("text") or "")),
            reverse=True,
        )[:8]

        topics_out.append(
            {
                "id": f"topic-{len(topics_out) + 1}",
                "label": label,
                "keywords": keywords,
                "item_count": len(member_indices),
                "share_pct": round(len(member_indices) * 100.0 / max(1, n_items), 1),
                "chunk_count": sum(1 for idx in member_indices if items[idx]["source_type"] == "document"),
                "memory_count": sum(1 for idx in member_indices if items[idx]["source_type"] == "memory"),
                "primary_source": primary_source,
                "sources": [
                    {"label": src, "count": cnt} for src, cnt in source_counter.most_common(6)
                ],
                "top_entities": top_entities,
                "representative_chunks": [
                    {
                        "id": it["id"],
                        "source_type": it["source_type"],
                        "source_label": it["source_label"],
                        "document_id": it.get("document_id"),
                        "version": it.get("version"),
                        "heading": it.get("heading") or "",
                        "preview": " ".join((it.get("text") or "").split())[:260],
                        "text": it.get("text") or "",
                        "entities": [e.get("name") for e in (it.get("entities") or [])[:5]],
                    }
                    for it in reps
                ],
            }
        )

    return {
        "total_items": n_items,
        "topic_count": len(topics_out),
        "topics": topics_out,
    }
