from pathlib import Path
from typing import Protocol
from memory_lab.core.models import Chunk, Document, Evidence, NormalizedDocument


class DocumentParser(Protocol):
    def parse(self, path: Path, document: Document) -> tuple[NormalizedDocument, dict]: ...


class ChunkingStrategy(Protocol):
    def chunk(self, document: NormalizedDocument, artifact: dict, run_id: str) -> list[Chunk]: ...


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorIndex(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]], namespace: str) -> None: ...
    def search(self, vector: list[float], limit: int, namespace: str) -> list[Evidence]: ...
