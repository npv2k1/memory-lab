from collections import Counter
import math
from memory_lab.core.models import Chunk, Evidence
from memory_lab.documents.enrichment.extractor import tokenize
from memory_lab.storage.filesystem.store import FileStore


class BM25Scorer:
    """Pure-Python Okapi BM25 scorer over arbitrary (id, text) pairs."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def score(self, query: str, documents: list[tuple[str, str]], limit: int = 10) -> list[tuple[str, float]]:
        query_terms = list(dict.fromkeys(tokenize(query)))
        if not query_terms or not documents:
            return []

        tokenized_docs: list[tuple[str, Counter[str], int]] = []
        doc_freq: Counter[str] = Counter()
        total_len = 0

        for doc_id, text in documents:
            tokens = tokenize(text)
            counts = Counter(tokens)
            length = len(tokens)
            total_len += length
            tokenized_docs.append((doc_id, counts, length))
            for term in counts:
                doc_freq[term] += 1

        n_docs = len(tokenized_docs)
        avgdl = total_len / n_docs if n_docs else 1.0
        if avgdl == 0:
            avgdl = 1.0

        scored: list[tuple[str, float]] = []
        for doc_id, counts, length in tokenized_docs:
            score = 0.0
            for term in query_terms:
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                df = doc_freq[term]
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (length / avgdl))
                score += idf * ((tf * (self.k1 + 1.0)) / denom)
            if score > 0.0:
                scored.append((doc_id, score))

        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:limit]


class BM25Index:
    """Filesystem-backed BM25 sparse index for document chunks."""

    def __init__(self, store: FileStore, *, k1: float = 1.5, b: float = 0.75):
        self.store = store
        self.scorer = BM25Scorer(k1=k1, b=b)

    def _key(self, namespace: str) -> str:
        return f"indexes/bm25/{namespace}.json"

    def _load(self, namespace: str) -> dict[str, dict]:
        key = self._key(namespace)
        if not self.store.path(key).exists():
            return {}
        return self.store.get(key)

    def upsert(self, chunks: list[Chunk], namespace: str) -> None:
        if not chunks:
            return
        with self.store.lock(f"bm25_{namespace}"):
            entries = self._load(namespace)
            for chunk in chunks:
                entries[chunk.id] = chunk.model_dump()
            self.store.put(self._key(namespace), entries)

    def search(
        self, query: str, limit: int, namespace: str, filters: dict[str, str] | None = None
    ) -> list[Evidence]:
        entries = self._load(namespace)
        if not entries:
            return []

        filtered_chunks: dict[str, Chunk] = {}
        for chunk_id, payload in entries.items():
            if filters and any(payload.get(field) != value for field, value in filters.items() if value):
                continue
            filtered_chunks[chunk_id] = Chunk.model_validate(payload)

        if not filtered_chunks:
            return []

        ranked = self.scorer.score(
            query,
            [(cid, chunk.text) for cid, chunk in filtered_chunks.items()],
            limit=limit,
        )
        results: list[Evidence] = []
        for chunk_id, score in ranked:
            chunk = filtered_chunks[chunk_id]
            results.append(
                Evidence(
                    chunk=chunk,
                    score=round(score, 6),
                    source_url=f"/documents/{chunk.document_id}/versions/{chunk.version}/source",
                    score_breakdown={"sparse_bm25": round(score, 6)},
                )
            )
        return results

    def delete_by_filter(self, filters: dict[str, str], namespace: str) -> int:
        active = {k: v for k, v in filters.items() if v}
        if not active:
            return 0
        with self.store.lock(f"bm25_{namespace}"):
            entries = self._load(namespace)
            if not entries:
                return 0
            kept: dict[str, dict] = {}
            removed = 0
            for chunk_id, payload in entries.items():
                if all(payload.get(field) == value for field, value in active.items()):
                    removed += 1
                else:
                    kept[chunk_id] = payload
            if removed > 0:
                self.store.put(self._key(namespace), kept)
            return removed
