from collections import Counter
import math
from typing import Any, Literal
import numpy as np
from memory_lab.core.models import Chunk, Document, EnrichmentArtifact
from memory_lab.documents.enrichment.extractor import STOPWORDS, fold_ascii, tokenize
from memory_lab.memory.service import MemoryService
from memory_lab.storage.filesystem.store import FileStore

AtlasSourceFilter = Literal["all", "documents", "memory"]


def _build_tfidf_matrix(texts: list[str], extra_terms: list[list[str]]) -> tuple[np.ndarray, list[str]]:
    tokenized: list[list[str]] = []
    df: Counter[str] = Counter()
    for text, extras in zip(texts, extra_terms, strict=True):
        toks = [
            t
            for t in tokenize(text, include_folded=False)
            if t not in STOPWORDS and fold_ascii(t) not in STOPWORDS and len(t) >= 2
        ]
        # Boost extracted entities/tags slightly
        boosted = toks + [e.lower() for e in extras for _ in range(2)]
        tokenized.append(boosted)
        for term in set(boosted):
            df[term] += 1

    vocab = [term for term, _ in df.most_common(512)]
    if not vocab:
        return np.zeros((len(texts), 1), dtype=np.float64), []
    term_to_idx = {term: i for i, term in enumerate(vocab)}
    n_docs = len(texts)
    mat = np.zeros((n_docs, len(vocab)), dtype=np.float64)
    for row_idx, toks in enumerate(tokenized):
        counts = Counter(toks)
        total = sum(counts.values()) or 1
        for term, cnt in counts.items():
            col = term_to_idx.get(term)
            if col is not None:
                idf = math.log((n_docs + 1) / (df[term] + 1)) + 1.0
                mat[row_idx, col] = (cnt / total) * idf
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return mat / norms, vocab


def _project_2d(features: np.ndarray) -> np.ndarray:
    n = features.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if n == 1:
        return np.array([[0.0, 0.0]], dtype=np.float64)
    if n == 2:
        return np.array([[-0.55, 0.0], [0.55, 0.0]], dtype=np.float64)

    centered = features - features.mean(axis=0, keepdims=True)
    try:
        u, s, _vt = np.linalg.svd(centered, full_matrices=False)
        if u.shape[1] >= 2:
            coords = u[:, :2] * s[:2]
        else:
            coords = np.column_stack([u[:, 0] * s[0], np.zeros(n)])
    except np.linalg.LinAlgError:
        angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
        coords = np.column_stack([np.cos(angles), np.sin(angles)])

    # Normalize initial PCA coords to [-0.88, 0.88]
    max_abs = np.max(np.abs(coords))
    if max_abs > 1e-9:
        coords = (coords / max_abs) * 0.88
    else:
        angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
        coords = np.column_stack([np.cos(angles) * 0.6, np.sin(angles) * 0.6])

    # Lightweight kNN attraction + collision separation so semantic clusters form crisp islands
    sim = np.clip(features @ features.T, 0.0, 1.0)
    np.fill_diagonal(sim, 0.0)
    k_pull = min(3, n - 1)
    for _ in range(18):
        disp = np.zeros_like(coords)
        # Pull toward top-k cosine neighbors
        for i in range(n):
            nbrs = np.argsort(sim[i])[::-1][:k_pull]
            for j in nbrs:
                w = float(sim[i, j])
                if w > 0.12:
                    disp[i] += (coords[j] - coords[i]) * (0.045 * w)
        # Gentle repulsion to avoid exact overlapping points
        for i in range(n):
            diff = coords[i] - coords
            dist_sq = np.sum(diff * diff, axis=1) + 1e-4
            dist_sq[i] = 1e9
            close = dist_sq < 0.09
            if np.any(close):
                push = diff[close] / dist_sq[close, None]
                disp[i] += np.sum(push, axis=0) * 0.0045
        coords = coords + np.clip(disp, -0.08, 0.08)

    max_abs = np.max(np.abs(coords))
    if max_abs > 1e-9:
        coords = (coords / max_abs) * 0.9
    return coords


def _kmeans(features: np.ndarray, coords: np.ndarray, k: int) -> np.ndarray:
    n = features.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    k = max(1, min(k, n))
    if k == 1:
        return np.zeros(n, dtype=np.int32)

    # Cluster on combined semantic features + 2D spatial layout for visual coherence
    joint = np.hstack([features * 0.75, coords * 0.25])
    # Deterministic farthest-point seeding (K-Means++ style without randomness)
    centers = [joint[0]]
    for _ in range(1, k):
        dists = np.min(
            np.stack([np.sum((joint - c) ** 2, axis=1) for c in centers], axis=0),
            axis=0,
        )
        centers.append(joint[int(np.argmax(dists))])
    centroids = np.stack(centers, axis=0)

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(20):
        dists = np.stack([np.sum((joint - c) ** 2, axis=1) for c in centroids], axis=1)
        new_labels = np.argmin(dists, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cid in range(k):
            mask = labels == cid
            if np.any(mask):
                centroids[cid] = joint[mask].mean(axis=0)
    return labels


def _compute_kde_grid(coords: np.ndarray, grid_size: int = 44, bound: float = 1.15) -> dict[str, Any]:
    if coords.shape[0] == 0:
        return {"grid_size": grid_size, "bound": bound, "values": [0.0] * (grid_size * grid_size)}
    axis = np.linspace(-bound, bound, grid_size)
    gx, gy = np.meshgrid(axis, axis)
    grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (G^2, 2)

    # Adaptive Silverman-inspired bandwidth
    bandwidth = max(0.16, min(0.36, 0.55 / (coords.shape[0] ** 0.22)))
    inv_2h2 = 1.0 / (2.0 * bandwidth * bandwidth)

    diff = grid_pts[:, None, :] - coords[None, :, :]  # (G^2, N, 2)
    dist_sq = np.sum(diff * diff, axis=2)
    density = np.sum(np.exp(-dist_sq * inv_2h2), axis=1)
    max_d = float(np.max(density))
    if max_d > 1e-9:
        density = density / max_d
    return {
        "grid_size": grid_size,
        "bound": bound,
        "values": [round(float(v), 4) for v in density],
    }


def _label_clusters(
    points: list[dict[str, Any]], coords: np.ndarray, labels: np.ndarray
) -> list[dict[str, Any]]:
    global_counts = Counter()
    for pt in points:
        candidates = pt["entities"] or pt["keywords"]
        for term in set(candidates):
            global_counts[term] += 1

    clusters: list[dict[str, Any]] = []
    unique_cids = sorted(set(int(c) for c in labels.tolist()))
    for cid in unique_cids:
        idxs = [i for i, c in enumerate(labels.tolist()) if c == cid]
        if not idxs:
            continue
        term_counts = Counter()
        for i in idxs:
            for ent in points[i]["entities"]:
                term_counts[ent] += 2
            for kw in points[i]["keywords"][:5]:
                if kw not in term_counts:
                    term_counts[kw] += 1

        # Rank by distinctiveness (cluster tf / global df)
        scored_terms = sorted(
            term_counts.items(),
            key=lambda kv: (-(kv[1] / (1.0 + 0.5 * global_counts.get(kv[0], 1))), -kv[1], kv[0]),
        )
        top_terms = [t for t, _ in scored_terms[:3]]
        label = " · ".join(top_terms) if top_terms else f"Cluster {cid + 1}"
        cx = float(np.mean(coords[idxs, 0]))
        cy = float(np.mean(coords[idxs, 1]))
        clusters.append(
            {
                "id": cid,
                "label": label,
                "keywords": top_terms,
                "count": len(idxs),
                "x": round(cx, 5),
                "y": round(cy, 5),
            }
        )
    return clusters


class EmbeddingAtlasService:
    """Computes Apple Embedding Atlas-style 2D projections, KDE density contours, clusters, kNN, and cross-filter facets."""

    def __init__(self, store: FileStore, memory: MemoryService):
        self.store = store
        self.memory = memory

    def _documents_map(self) -> dict[tuple[str, str], Document]:
        out: dict[tuple[str, str], Document] = {}
        for raw in self.store.documents():
            doc = Document.model_validate(raw)
            out[(doc.id, doc.version)] = doc
        return out

    def build_atlas(
        self,
        *,
        source: AtlasSourceFilter = "all",
        n_clusters: int | None = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        docs_map = self._documents_map()
        raw_items: list[dict[str, Any]] = []
        dense_vectors: list[list[float] | None] = []

        # 1. Load Document chunks + cached embeddings + enrichments
        if source in ("all", "documents"):
            enriched_by_chunk: dict[str, dict[str, Any]] = {}
            for path in sorted(self.store.root.glob("runs/*/enriched.json")):
                try:
                    art = EnrichmentArtifact.model_validate(self.store.get(str(path.relative_to(self.store.root))))
                    for ec in art.chunks:
                        enriched_by_chunk[ec.chunk_id] = {
                            "entities": [e.name for e in ec.entities],
                            "temporal": [t.normalized for t in ec.temporal],
                            "keywords": ec.keywords,
                        }
                except Exception:
                    continue

            seen_chunks: set[str] = set()
            for run_dir in sorted(self.store.root.glob("runs/*")):
                chunks_path = run_dir / "chunks.json"
                if not chunks_path.exists():
                    continue
                try:
                    chunk_rows = self.store.get(str(chunks_path.relative_to(self.store.root)))
                    vectors_path = run_dir / "vectors.json"
                    vecs = (
                        self.store.get(str(vectors_path.relative_to(self.store.root)))
                        if vectors_path.exists()
                        else None
                    )
                except Exception:
                    continue

                for idx, row in enumerate(chunk_rows):
                    chunk = Chunk.model_validate(row)
                    if chunk.id in seen_chunks:
                        continue
                    seen_chunks.add(chunk.id)
                    doc = docs_map.get((chunk.document_id, chunk.version))
                    fname = doc.filename if doc else f"doc:{chunk.document_id[:8]}"
                    meta = enriched_by_chunk.get(chunk.id, {"entities": [], "temporal": [], "keywords": []})
                    date_val = meta["temporal"][0][:10] if meta["temporal"] else "undated"
                    raw_items.append(
                        {
                            "id": f"chunk:{chunk.id}",
                            "ref_id": chunk.id,
                            "source_type": "document",
                            "title": f"{fname} · {', '.join(chunk.block_ids[:2])}",
                            "text": chunk.text,
                            "preview": chunk.text[:220],
                            "strategy": "document_chunk",
                            "session_id": fname,
                            "date": date_val,
                            "entities": meta["entities"],
                            "keywords": meta["keywords"],
                            "tags": [],
                            "block_ids": chunk.block_ids,
                            "source_url": f"/documents/{chunk.document_id}/versions/{chunk.version}/source",
                        }
                    )
                    dense_vectors.append(vecs[idx] if isinstance(vecs, list) and idx < len(vecs) else None)

        # 2. Load Agent Memory records + cached vectors
        if source in ("all", "memory"):
            mem_ns = self.memory.vector.memory_namespace
            for rec in self.memory.list_records(limit=limit):
                vec_key = f"memory/vectors/{mem_ns}/{rec.id}.json"
                vec = self.store.get(vec_key) if self.store.path(vec_key).exists() else None
                raw_items.append(
                    {
                        "id": f"mem:{rec.id}",
                        "ref_id": rec.id,
                        "source_type": "memory",
                        "title": f"{rec.session_id} · {rec.strategy}",
                        "text": rec.content,
                        "preview": rec.summary or rec.content[:220],
                        "strategy": rec.strategy,
                        "session_id": rec.session_id,
                        "date": rec.timestamp[:10],
                        "entities": [e.name for e in rec.entities],
                        "keywords": rec.keywords,
                        "tags": rec.tags,
                        "block_ids": [],
                        "source_url": None,
                    }
                )
                dense_vectors.append(vec)

        raw_items = raw_items[:limit]
        dense_vectors = dense_vectors[:limit]

        if not raw_items:
            return {
                "source": source,
                "projection": "svd-knn",
                "used_dense_embeddings": False,
                "total_points": 0,
                "points": [],
                "clusters": [],
                "density": _compute_kde_grid(np.zeros((0, 2))),
                "facets": {"sources": [], "clusters": [], "strategies": [], "dates": [], "entities": []},
            }

        # 3. Build unified semantic feature matrix (Dense embeddings + TF-IDF/Entity features)
        lex_mat, _vocab = _build_tfidf_matrix(
            [item["text"] for item in raw_items],
            [item["entities"] + item["tags"] for item in raw_items],
        )

        valid_dims = {len(v) for v in dense_vectors if isinstance(v, list) and len(v) > 0}
        used_dense = False
        if len(valid_dims) == 1:
            dim = next(iter(valid_dims))
            dense_mat = np.zeros((len(raw_items), dim), dtype=np.float64)
            present_count = 0
            for idx, v in enumerate(dense_vectors):
                if isinstance(v, list) and len(v) == dim:
                    arr = np.asarray(v, dtype=np.float64)
                    norm = float(np.linalg.norm(arr))
                    if norm > 0:
                        dense_mat[idx] = arr / norm
                        present_count += 1
            if present_count > 0:
                used_dense = True
                # Combine normalized dense + lexical/entity representations
                features = np.hstack([dense_mat * 0.65, lex_mat * 0.75])
                norms = np.linalg.norm(features, axis=1, keepdims=True)
                features = features / np.where(norms == 0, 1.0, norms)
            else:
                features = lex_mat
        else:
            features = lex_mat

        # 4. 2D Projection, Automatic Clustering + Labels, KDE Density Grid, Top-5 kNN
        coords = _project_2d(features)
        n_pts = len(raw_items)
        k_clusters = n_clusters if n_clusters is not None else (
            max(2, min(7, round(math.sqrt(n_pts / 1.5)))) if n_pts >= 4 else max(1, n_pts)
        )
        cluster_ids = _kmeans(features, coords, k_clusters)
        clusters = _label_clusters(raw_items, coords, cluster_ids)
        cluster_label_map = {c["id"]: c["label"] for c in clusters}
        density = _compute_kde_grid(coords)

        sim_matrix = np.clip(features @ features.T, -1.0, 1.0)
        np.fill_diagonal(sim_matrix, -2.0)

        points: list[dict[str, Any]] = []
        for i, item in enumerate(raw_items):
            cid = int(cluster_ids[i])
            nbr_indices = np.argsort(sim_matrix[i])[::-1][: min(5, n_pts - 1)]
            neighbors = [
                {
                    "id": raw_items[int(j)]["id"],
                    "title": raw_items[int(j)]["title"],
                    "score": round(float(max(0.0, sim_matrix[i, int(j)])), 4),
                }
                for j in nbr_indices
                if sim_matrix[i, int(j)] > 0.0
            ]
            points.append(
                {
                    **item,
                    "x": round(float(coords[i, 0]), 5),
                    "y": round(float(coords[i, 1]), 5),
                    "cluster_id": cid,
                    "cluster_label": cluster_label_map.get(cid, f"Cluster {cid + 1}"),
                    "has_dense_vector": dense_vectors[i] is not None,
                    "neighbors": neighbors,
                }
            )

        # 5. Facet summaries for Apple Embedding Atlas-style linked cross-filter charts
        source_counts = Counter(p["source_type"] for p in points)
        strategy_counts = Counter(p["strategy"] for p in points)
        date_counts = Counter(p["date"] for p in points)
        entity_counts = Counter(e for p in points for e in p["entities"])

        return {
            "source": source,
            "projection": "svd-knn",
            "used_dense_embeddings": used_dense,
            "total_points": len(points),
            "points": points,
            "clusters": clusters,
            "density": density,
            "facets": {
                "sources": [{"label": k, "count": v} for k, v in source_counts.most_common()],
                "clusters": [{"id": c["id"], "label": c["label"], "count": c["count"]} for c in clusters],
                "strategies": [{"label": k, "count": v} for k, v in strategy_counts.most_common()],
                "dates": [{"label": k, "count": v} for k, v in sorted(date_counts.items())],
                "entities": [{"label": k, "count": v} for k, v in entity_counts.most_common(12)],
            },
        }
