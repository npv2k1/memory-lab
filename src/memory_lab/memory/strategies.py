import logging
import math
from typing import Any
from uuid import uuid4
from memory_lab.core.contracts import EmbeddingProvider, GraphStore
from memory_lab.core.models import (
    InteractionTrace,
    MemoryRecord,
    MemorySearchHit,
    MemoryStrategyName,
)
from memory_lab.documents.enrichment.extractor import NonGenerativeExtractor, normalize_name, tokenize
from memory_lab.memory.temporal import TemporalHierarchy, record_matches_bucket
from memory_lab.retrieval.fusion import reciprocal_rank_fusion
from memory_lab.retrieval.sparse import BM25Scorer
from memory_lab.storage.filesystem.store import FileStore, digest

logger = logging.getLogger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _build_summary(text: str, max_len: int = 180) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 1].rstrip() + "…"


class MemoryRecordStore:
    """Filesystem source-of-truth repository for MemoryRecord artifacts."""

    def __init__(self, store: FileStore):
        self.store = store

    def _key(self, memory_id: str) -> str:
        return f"memory/records/{memory_id}.json"

    def save(self, record: MemoryRecord) -> MemoryRecord:
        self.store.put(self._key(record.id), record.model_dump())
        return record

    def get(
        self,
        memory_id: str,
        *,
        include_deleted: bool = False,
        include_archived: bool = False,
    ) -> MemoryRecord:
        record = MemoryRecord.model_validate(self.store.get(self._key(memory_id)))
        if record.deleted and not include_deleted:
            raise FileNotFoundError(f"Memory not found: {memory_id}")
        if record.archived and not include_archived:
            raise FileNotFoundError(f"Memory archived: {memory_id}")
        return record

    def list(
        self,
        *,
        include_deleted: bool = False,
        include_archived: bool = False,
        session_id: str | None = None,
        project: str | None = None,
        temporal_bucket: str | None = None,
    ) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        for path in self.store.root.glob("memory/records/*.json"):
            record = MemoryRecord.model_validate(self.store.get(str(path.relative_to(self.store.root))))
            if record.deleted and not include_deleted:
                continue
            if record.archived and not include_archived:
                continue
            if session_id and record.session_id != session_id:
                continue
            if project and record.project != project:
                continue
            if temporal_bucket and not record_matches_bucket(record, temporal_bucket):
                continue
            records.append(record)
        records.sort(key=lambda item: (item.timestamp, item.id), reverse=True)
        return records

    def bump_access(self, memory_id: str, *, cooldown_secs: int = 60) -> MemoryRecord | None:
        """Throttled access-weighted reinforcement (ai-memory M8): never rewrites search indexes."""
        from datetime import datetime, timezone
        from memory_lab.core.models import utc_now_iso

        key = self._key(memory_id)
        if not self.store.path(key).exists():
            return None
        with self.store.lock(f"mem_access_{memory_id}"):
            record = MemoryRecord.model_validate(self.store.get(key))
            if record.deleted or record.archived:
                return None
            now = datetime.now(timezone.utc)
            if record.last_accessed_at and cooldown_secs > 0:
                try:
                    prev = datetime.fromisoformat(record.last_accessed_at.replace("Z", "+00:00"))
                    if prev.tzinfo is None:
                        prev = prev.replace(tzinfo=timezone.utc)
                    if (now - prev).total_seconds() < cooldown_secs:
                        return record
                except ValueError:
                    pass
            record.access_count += 1
            record.last_accessed_at = utc_now_iso()
            self.save(record)
            return record

    def mark_deleted(self, memory_id: str) -> bool:
        key = self._key(memory_id)
        if not self.store.path(key).exists():
            return False
        record = MemoryRecord.model_validate(self.store.get(key))
        if record.deleted:
            return False
        record.deleted = True
        self.save(record)
        return True


class _BaseMemoryStrategy:
    name: MemoryStrategyName

    def __init__(
        self,
        file_store: FileStore,
        records: MemoryRecordStore,
        extractor: NonGenerativeExtractor,
        graph: GraphStore,
        temporal: TemporalHierarchy,
        namespace: str,
    ):
        self.file_store = file_store
        self.records = records
        self.extractor = extractor
        self.graph = graph
        self.temporal = temporal
        self.namespace = namespace

    def _create_record(
        self,
        trace: InteractionTrace,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        record_id = str(uuid4())
        merged_meta = {**trace.metadata, **dict(metadata or {})}
        project = str(merged_meta.pop("_project", None) or "default")
        tier = str(merged_meta.pop("_tier", None) or "episodic")
        pinned = bool(merged_meta.pop("_pinned", False))
        expires_at = merged_meta.pop("_expires_at", None)

        entities, relations, temporal, keywords = self.extractor.extract_text(
            trace.content, evidence_id=record_id, reference_timestamp=trace.timestamp
        )
        record = MemoryRecord(
            id=record_id,
            strategy=self.name,
            namespace=self.namespace,
            project=project,
            tier=tier,  # type: ignore[arg-type]
            pinned=pinned,
            expires_at=expires_at,
            content=trace.content,
            summary=_build_summary(trace.content),
            trace_ids=[trace.id],
            session_id=trace.session_id,
            role=trace.role,
            timestamp=trace.timestamp,
            entities=entities,
            relations=relations,
            temporal=temporal,
            keywords=keywords,
            hierarchy_level="raw",
            tags=list(tags or []),
            source_refs=list(trace.source_refs),
            metadata=merged_meta,
        )
        self.records.save(record)
        self.graph.upsert_record(record)
        self.temporal.rebuild(self.records.list())
        return record

    def get(self, memory_id: str) -> MemoryRecord:
        return self.records.get(memory_id)

    def delete(self, memory_id: str) -> bool:
        deleted = self.records.mark_deleted(memory_id)
        if deleted:
            self.graph.delete_record(memory_id)
            self.temporal.rebuild(self.records.list())
        return deleted


class VectorMemory(_BaseMemoryStrategy):
    """Dense embedding memory backed by filesystem vector cache and Qdrant (with cosine fallback)."""

    name: MemoryStrategyName = "vector"

    def __init__(
        self,
        file_store: FileStore,
        records: MemoryRecordStore,
        extractor: NonGenerativeExtractor,
        graph: GraphStore,
        temporal: TemporalHierarchy,
        embedding: EmbeddingProvider,
        index: Any,
        embedding_identity: str,
    ):
        self.memory_namespace = f"mem_{digest(embedding_identity.encode())[:16]}"
        super().__init__(file_store, records, extractor, graph, temporal, self.memory_namespace)
        self.embedding = embedding
        self.index = index

    def _vector_key(self, record_id: str) -> str:
        return f"memory/vectors/{self.memory_namespace}/{record_id}.json"

    def index_record(self, record: MemoryRecord) -> list[float]:
        key = self._vector_key(record.id)
        if self.file_store.path(key).exists():
            vector = self.file_store.get(key)
        else:
            vector = self.embedding.embed([record.content])[0]
            self.file_store.put(key, vector)
        if self.index is not None and hasattr(self.index, "upsert_points"):
            payload = {
                "record_id": record.id,
                "session_id": record.session_id,
                "strategy": record.strategy,
                "timestamp": record.timestamp,
            }
            self.index.upsert_points([record.id], [vector], [payload], self.memory_namespace)
        return vector

    def store(
        self,
        trace: InteractionTrace,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        record = self._create_record(trace, tags=tags, metadata=metadata)
        self.index_record(record)
        return record

    def search_scored(
        self,
        query: str,
        *,
        limit: int = 5,
        session_id: str | None = None,
        temporal_bucket: str | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        query_vector = self.embedding.embed([query])[0]
        filters = {"session_id": session_id} if session_id else None

        candidates: list[tuple[str, float]] = []
        if self.index is not None and hasattr(self.index, "search_points"):
            hits = self.index.search_points(
                query_vector, max(limit * 4, 20), self.memory_namespace, filters=filters
            )
            candidates = [(pid, score) for pid, score, _payload in hits]
        else:
            # Filesystem cosine fallback when no vector store with point API is attached
            for record in self.records.list(session_id=session_id, temporal_bucket=temporal_bucket):
                key = self._vector_key(record.id)
                if self.file_store.path(key).exists():
                    vec = self.file_store.get(key)
                    candidates.append((record.id, _cosine(query_vector, vec)))
            candidates.sort(key=lambda item: (-item[1], item[0]))

        results: list[tuple[MemoryRecord, float]] = []
        for record_id, score in candidates:
            try:
                record = self.records.get(record_id)
            except FileNotFoundError:
                continue
            if session_id and record.session_id != session_id:
                continue
            if temporal_bucket and not record_matches_bucket(record, temporal_bucket):
                continue
            results.append((record, round(float(score), 6)))
            if len(results) >= limit:
                break
        return results

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        session_id: str | None = None,
        temporal_bucket: str | None = None,
    ) -> list[MemorySearchHit]:
        scored = self.search_scored(
            query, limit=limit, session_id=session_id, temporal_bucket=temporal_bucket
        )
        return [
            MemorySearchHit(
                record=record,
                score=score,
                strategy=self.name,
                score_breakdown={"dense": score},
                evidence_trace_ids=record.trace_ids,
            )
            for record, score in scored
        ]

    def delete(self, memory_id: str) -> bool:
        deleted = super().delete(memory_id)
        if deleted and self.index is not None and hasattr(self.index, "delete_points"):
            try:
                self.index.delete_points([memory_id], self.memory_namespace)
            except Exception:
                logger.warning("memory_vector_delete_failed", extra={"memory_id": memory_id})
        return deleted


_PREDICATE_HINTS: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("manager", "boss", "report", "reports", "lead", "leader", "director", "quản lý", "quan ly", "sếp", "sep", "báo cáo", "bao cao", "trưởng nhóm", "truong nhom"), ("reports_to", "manages")),
    (("project", "initiative", "dự án", "du an"), ("manages", "works_on")),
    (("work", "works", "company", "employer", "làm việc", "lam viec", "công ty", "cong ty"), ("works_at", "works_on")),
    (("deploy", "deployed", "host", "hosted", "server", "triển khai", "trien khai", "chạy trên", "chay tren"), ("deployed_on",)),
    (("prefer", "prefers", "like", "likes", "favorite", "thích", "thich", "ưa thích"), ("prefers",)),
    (("use", "uses", "using", "tool", "stack", "sử dụng", "su dung", "dùng", "dung"), ("uses", "depends_on")),
    (("connect", "integrate", "kết nối", "ket noi"), ("integrates_with",)),
    (("store", "stores", "save", "lưu", "luu"), ("stores",)),
    (("create", "created", "build", "built", "author", "tạo", "xây dựng"), ("created",)),
]


def _infer_predicate_hints(query: str) -> list[str]:
    lower = query.lower()
    predicates: list[str] = []
    for cues, preds in _PREDICATE_HINTS:
        if any(cue in lower for cue in cues):
            for p in preds:
                if p not in predicates:
                    predicates.append(p)
    return predicates


class GraphMemory(_BaseMemoryStrategy):
    """Entity and relation graph memory with 1-hop and 2-hop neighborhood retrieval."""

    name: MemoryStrategyName = "graph"

    def store(
        self,
        trace: InteractionTrace,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        return self._create_record(trace, tags=tags, metadata=metadata)

    def search_scored(
        self,
        query: str,
        *,
        limit: int = 5,
        session_id: str | None = None,
        temporal_bucket: str | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        entities = [e.normalized_name for e in self.extractor.extract_entities(query)]
        keywords = self.extractor.extract_keywords(query, limit=10)
        tokens = [tok for tok in tokenize(query, include_folded=True) if len(tok) >= 3]
        query_terms = list(dict.fromkeys([*entities, *keywords, *tokens]))
        predicates = _infer_predicate_hints(query)

        try:
            raw_hits = self.graph.query_by_entities(
                query_terms, limit=max(limit * 4, 20), predicates=predicates  # type: ignore[call-arg]
            )
        except TypeError:
            raw_hits = self.graph.query_by_entities(query_terms, limit=max(limit * 4, 20))

        results: list[tuple[MemoryRecord, float]] = []
        for record_id, score in raw_hits:
            try:
                record = self.records.get(record_id)
            except FileNotFoundError:
                continue
            if session_id and record.session_id != session_id:
                continue
            if temporal_bucket and not record_matches_bucket(record, temporal_bucket):
                continue
            results.append((record, round(float(score), 6)))
            if len(results) >= limit:
                break

        # Fallback: when graph nodes have zero direct entity matches (e.g. lowercase/natural-language query),
        # rank records via BM25 over entity/relation/keyword/content representations so graph search stays resilient.
        if not results:
            candidates = self.records.list(session_id=session_id, temporal_bucket=temporal_bucket)
            if candidates:
                scorer = BM25Scorer()
                docs = [
                    (
                        r.id,
                        " ".join(
                            [
                                *(e.name for e in r.entities),
                                *(f"{rel.subject} {rel.predicate} {rel.object}" for rel in r.relations),
                                *r.keywords,
                                r.content,
                            ]
                        ),
                    )
                    for r in candidates
                ]
                by_id = {r.id: r for r in candidates}
                for rid, score in scorer.score(query, docs, limit=limit):
                    rec = by_id.get(rid)
                    if rec is not None:
                        results.append((rec, round(float(score) * 0.25, 6)))
        return results

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        session_id: str | None = None,
        temporal_bucket: str | None = None,
    ) -> list[MemorySearchHit]:
        scored = self.search_scored(
            query, limit=limit, session_id=session_id, temporal_bucket=temporal_bucket
        )
        return [
            MemorySearchHit(
                record=record,
                score=score,
                strategy=self.name,
                score_breakdown={"graph": score},
                evidence_trace_ids=record.trace_ids,
            )
            for record, score in scored
        ]


class ZeroTokenMemory(_BaseMemoryStrategy):
    """Zero-LLM, zero-embedding-token memory using BM25 + entity/tag overlap + temporal hierarchy."""

    name: MemoryStrategyName = "zero_token"

    def __init__(
        self,
        store: FileStore,
        records: MemoryRecordStore,
        extractor: NonGenerativeExtractor,
        graph: GraphStore,
        temporal: TemporalHierarchy,
        namespace: str,
    ):
        super().__init__(store, records, extractor, graph, temporal, namespace)
        self.bm25 = BM25Scorer()

    def store(
        self,
        trace: InteractionTrace,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        return self._create_record(trace, tags=tags, metadata=metadata)

    def search_signals(
        self,
        query: str,
        *,
        limit: int = 5,
        session_id: str | None = None,
        temporal_bucket: str | None = None,
    ) -> tuple[dict[str, MemoryRecord], list[tuple[str, float]], list[tuple[str, float]], list[tuple[str, float]]]:
        candidates = self.records.list(session_id=session_id, temporal_bucket=temporal_bucket)
        by_id = {r.id: r for r in candidates}
        if not candidates:
            return by_id, [], [], []

        candidate_limit = max(limit * 4, 20)
        # 1. BM25 lexical signal over content + tags + entity names
        bm25_docs = [
            (
                r.id,
                " ".join([r.content, *r.tags, *(e.name for e in r.entities)]),
            )
            for r in candidates
        ]
        bm25_ranked = self.bm25.score(query, bm25_docs, limit=candidate_limit)

        # 2. Non-generative entity/keyword overlap signal
        query_entities = {e.normalized_name for e in self.extractor.extract_entities(query)}
        query_tokens = set(tokenize(query))
        entity_ranked: list[tuple[str, float]] = []
        for record in candidates:
            rec_entities = {e.normalized_name for e in record.entities}
            rec_tags = {normalize_name(t) for t in record.tags}
            overlap = len(query_entities & rec_entities) * 1.5 + len(query_tokens & (rec_entities | rec_tags)) * 0.6
            if overlap > 0:
                entity_ranked.append((record.id, round(overlap, 6)))
        entity_ranked.sort(key=lambda item: (-item[1], item[0]))
        entity_ranked = entity_ranked[:candidate_limit]

        # 3. Temporal signal
        query_temporal = self.extractor.extract_temporal(query)
        temporal_ranked = self.temporal.score_records(
            candidates, query_temporal=query_temporal, temporal_bucket=temporal_bucket
        )[:candidate_limit]

        return by_id, bm25_ranked, entity_ranked, temporal_ranked

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        session_id: str | None = None,
        temporal_bucket: str | None = None,
    ) -> list[MemorySearchHit]:
        by_id, bm25_ranked, entity_ranked, temporal_ranked = self.search_signals(
            query, limit=limit, session_id=session_id, temporal_bucket=temporal_bucket
        )
        fused = reciprocal_rank_fusion(
            {
                "sparse_bm25": bm25_ranked,
                "entity_overlap": entity_ranked,
                "temporal": temporal_ranked,
            },
            weights={"sparse_bm25": 1.2, "entity_overlap": 0.9, "temporal": 0.7},
        )
        hits: list[MemorySearchHit] = []
        for record_id, score, breakdown in fused[:limit]:
            record = by_id.get(record_id)
            if not record:
                continue
            hits.append(
                MemorySearchHit(
                    record=record,
                    score=score,
                    strategy=self.name,
                    score_breakdown=breakdown,
                    evidence_trace_ids=record.trace_ids,
                )
            )
        return hits


class HybridMemory(_BaseMemoryStrategy):
    """Fuses VectorMemory (dense), GraphMemory (relational), BM25 (sparse), and TemporalHierarchy via RRF."""

    name: MemoryStrategyName = "hybrid"

    def __init__(
        self,
        store: FileStore,
        records: MemoryRecordStore,
        extractor: NonGenerativeExtractor,
        graph: GraphStore,
        temporal: TemporalHierarchy,
        vector_strategy: VectorMemory,
        graph_strategy: GraphMemory,
        zero_token_strategy: ZeroTokenMemory,
        namespace: str,
    ):
        super().__init__(store, records, extractor, graph, temporal, namespace)
        self.vector_strategy = vector_strategy
        self.graph_strategy = graph_strategy
        self.zero_token_strategy = zero_token_strategy

    def store(
        self,
        trace: InteractionTrace,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        record = self._create_record(trace, tags=tags, metadata=metadata)
        try:
            self.vector_strategy.index_record(record)
        except Exception:
            logger.warning("hybrid_memory_vector_index_skipped", extra={"memory_id": record.id})
        return record

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        session_id: str | None = None,
        temporal_bucket: str | None = None,
    ) -> list[MemorySearchHit]:
        candidate_limit = max(limit * 4, 20)
        by_id, bm25_ranked, _entity_ranked, temporal_ranked = self.zero_token_strategy.search_signals(
            query, limit=candidate_limit, session_id=session_id, temporal_bucket=temporal_bucket
        )
        graph_hits = self.graph_strategy.search_scored(
            query, limit=candidate_limit, session_id=session_id, temporal_bucket=temporal_bucket
        )
        graph_ranked = [(r.id, s) for r, s in graph_hits]
        for r, _ in graph_hits:
            by_id.setdefault(r.id, r)

        dense_ranked: list[tuple[str, float]] = []
        try:
            dense_hits = self.vector_strategy.search_scored(
                query, limit=candidate_limit, session_id=session_id, temporal_bucket=temporal_bucket
            )
            dense_ranked = [(r.id, s) for r, s in dense_hits]
            for r, _ in dense_hits:
                by_id.setdefault(r.id, r)
        except Exception:
            logger.warning("hybrid_memory_dense_search_skipped")

        fused = reciprocal_rank_fusion(
            {
                "dense": dense_ranked,
                "sparse_bm25": bm25_ranked,
                "graph": graph_ranked,
                "temporal": temporal_ranked,
            },
            weights={"dense": 1.2, "sparse_bm25": 1.0, "graph": 0.9, "temporal": 0.6},
        )
        hits: list[MemorySearchHit] = []
        for record_id, score, breakdown in fused[:limit]:
            record = by_id.get(record_id)
            if not record:
                continue
            hits.append(
                MemorySearchHit(
                    record=record,
                    score=score,
                    strategy=self.name,
                    score_breakdown=breakdown,
                    evidence_trace_ids=record.trace_ids,
                )
            )
        return hits

    def delete(self, memory_id: str) -> bool:
        return self.vector_strategy.delete(memory_id)
