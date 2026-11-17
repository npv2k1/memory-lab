from pathlib import Path
from typing import Protocol
from memory_lab.core.models import Chunk, Evidence, NormalizedDocument, ParsedDocument


class DocumentParser(Protocol):
    @property
    def version(self) -> str: ...
    def parse(self, path: Path) -> ParsedDocument: ...


class ChunkingStrategy(Protocol):
    def chunk(self, document: NormalizedDocument, artifact: dict, run_id: str) -> list[Chunk]: ...


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorIndex(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]], namespace: str) -> None: ...
    def search(self, vector: list[float], limit: int, namespace: str) -> list[Evidence]: ...
