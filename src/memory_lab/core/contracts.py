from pathlib import Path
from typing import Any, Protocol
from memory_lab.core.models import (
    Chunk,
    EnrichmentArtifact,
    Evidence,
    InteractionTrace,
    MemoryRecord,
    MemorySearchHit,
    NormalizedDocument,
    ParsedDocument,
)


class DocumentParser(Protocol):
    @property
    def version(self) -> str: ...
    def parse(self, path: Path) -> ParsedDocument: ...


class ChunkingStrategy(Protocol):
    def chunk(self, document: NormalizedDocument, artifact: dict, run_id: str) -> list[Chunk]: ...


class EnrichmentExtractor(Protocol):
    def extract_chunks(
        self, chunks: list[Chunk], *, pipeline_run_id: str, document_id: str, version: str
    ) -> EnrichmentArtifact: ...


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorIndex(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]], namespace: str) -> None: ...
    def search(
        self, vector: list[float], limit: int, namespace: str, filters: dict[str, str] | None = None
    ) -> list[Evidence]: ...


class SparseIndex(Protocol):
    def upsert(self, chunks: list[Chunk], namespace: str) -> None: ...
    def search(
        self, query: str, limit: int, namespace: str, filters: dict[str, str] | None = None
    ) -> list[Evidence]: ...


class GraphStore(Protocol):
    def upsert_record(self, record: MemoryRecord) -> None: ...
    def delete_record(self, record_id: str) -> None: ...
    def query_by_entities(self, entities: list[str], limit: int = 10) -> list[tuple[str, float]]: ...
    def snapshot(self, entity: str | None = None) -> dict[str, Any]: ...


class MemoryStrategy(Protocol):
    name: str

    def store(
        self,
        trace: InteractionTrace,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord: ...

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        session_id: str | None = None,
        temporal_bucket: str | None = None,
    ) -> list[MemorySearchHit]: ...

    def get(self, memory_id: str) -> MemoryRecord: ...
    def delete(self, memory_id: str) -> bool: ...
