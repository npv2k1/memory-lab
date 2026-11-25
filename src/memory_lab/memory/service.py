from typing import Any
from memory_lab.core.contracts import EmbeddingProvider, GraphStore, MemoryStrategy
from memory_lab.core.models import (
    InteractionTrace,
    MemoryRecord,
    MemorySearchHit,
    MemoryStrategyName,
)
from memory_lab.documents.enrichment.extractor import NonGenerativeExtractor
from memory_lab.memory.strategies import (
    GraphMemory,
    HybridMemory,
    MemoryRecordStore,
    VectorMemory,
    ZeroTokenMemory,
)
from memory_lab.memory.temporal import TemporalHierarchy
from memory_lab.memory.traces import TraceStore
from memory_lab.storage.filesystem.store import FileStore
from memory_lab.storage.graph.store import FileGraphStore


class MemoryService:
    """Unified service managing interaction traces, memory strategies, graph, and temporal hierarchy."""

    def __init__(
        self,
        store: FileStore,
        embedding: EmbeddingProvider,
        index: Any,
        embedding_identity: str,
        *,
        graph: GraphStore | None = None,
        extractor: NonGenerativeExtractor | None = None,
        default_strategy: MemoryStrategyName = "hybrid",
    ):
        self.store = store
        self.traces = TraceStore(store)
        self.records = MemoryRecordStore(store)
        self.extractor = extractor or NonGenerativeExtractor()
        self.graph: GraphStore = graph or FileGraphStore(store)
        self.temporal = TemporalHierarchy(store)
        self.default_strategy: MemoryStrategyName = default_strategy

        self.vector = VectorMemory(
            store, self.records, self.extractor, self.graph, self.temporal, embedding, index, embedding_identity
        )
        self.graph_strategy = GraphMemory(
            store, self.records, self.extractor, self.graph, self.temporal, self.vector.memory_namespace
        )
        self.zero_token = ZeroTokenMemory(
            store, self.records, self.extractor, self.graph, self.temporal, self.vector.memory_namespace
        )
        self.hybrid = HybridMemory(
            store,
            self.records,
            self.extractor,
            self.graph,
            self.temporal,
            self.vector,
            self.graph_strategy,
            self.zero_token,
            self.vector.memory_namespace,
        )
        self.strategies: dict[str, MemoryStrategy] = {
            "vector": self.vector,
            "graph": self.graph_strategy,
            "zero_token": self.zero_token,
            "hybrid": self.hybrid,
        }

    def configure_embedding(self, embedding: EmbeddingProvider, embedding_identity: str) -> None:
        """Hot-swap the active embedding provider and memory vector namespace at runtime."""
        from memory_lab.storage.filesystem.store import digest

        new_ns = f"mem_{digest(embedding_identity.encode())[:16]}"
        self.vector.embedding = embedding
        self.vector.memory_namespace = new_ns
        self.vector.namespace = new_ns
        self.graph_strategy.namespace = new_ns
        self.zero_token.namespace = new_ns
        self.hybrid.namespace = new_ns

    def _resolve_strategy(self, strategy: str | None) -> MemoryStrategy:
        name = strategy or self.default_strategy
        if name not in self.strategies:
            raise ValueError(f"Unsupported memory strategy: {name}")
        return self.strategies[name]

    def store_memory(
        self,
        *,
        content: str,
        session_id: str = "default",
        role: str = "user",
        strategy: MemoryStrategyName | None = None,
        timestamp: str | None = None,
        tags: list[str] | None = None,
        source_refs: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        if not content.strip():
            raise ValueError("Memory content must not be empty")
        trace = self.traces.append(
            content=content,
            session_id=session_id,
            role=role,
            timestamp=timestamp,
            source_refs=source_refs,
            metadata=metadata,
        )
        return self._resolve_strategy(strategy).store(trace, tags=tags, metadata=metadata)

    def search_memory(
        self,
        query: str,
        *,
        strategy: MemoryStrategyName | None = None,
        limit: int = 5,
        session_id: str | None = None,
        temporal_bucket: str | None = None,
    ) -> list[MemorySearchHit]:
        return self._resolve_strategy(strategy).search(
            query, limit=limit, session_id=session_id, temporal_bucket=temporal_bucket
        )

    def get_memory(self, memory_id: str) -> MemoryRecord:
        return self.records.get(memory_id)

    def delete_memory(self, memory_id: str) -> bool:
        return self.hybrid.delete(memory_id)

    def list_records(
        self,
        *,
        session_id: str | None = None,
        temporal_bucket: str | None = None,
        limit: int = 100,
    ) -> list[MemoryRecord]:
        return self.records.list(session_id=session_id, temporal_bucket=temporal_bucket)[:limit]

    def list_traces(self, *, session_id: str | None = None, limit: int = 100) -> list[InteractionTrace]:
        return self.traces.list(session_id=session_id, limit=limit)

    def temporal_hierarchy(self) -> dict[str, list[dict]]:
        return self.temporal.load()

    def graph_snapshot(self, entity: str | None = None) -> dict[str, Any]:
        return self.graph.snapshot(entity=entity)
