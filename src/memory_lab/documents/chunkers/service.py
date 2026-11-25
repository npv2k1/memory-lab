from collections.abc import Callable
from memory_lab.core.contracts import ChunkingStrategy
from memory_lab.core.models import Chunk, NormalizedDocument, PipelineConfig
from memory_lab.documents.chunkers.strategies import FixedChunker, HybridChunkerAdapter
from memory_lab.storage.filesystem.store import FileStore


class ChunkingService:
    def __init__(self, store: FileStore, strategies: dict[str, Callable[[PipelineConfig], ChunkingStrategy]] | None = None):
        self.store = store
        self.strategies = strategies if strategies is not None else {
            "fixed": lambda config: FixedChunker(config.chunk_size, config.overlap),
            "hybrid": lambda config: HybridChunkerAdapter(config.chunk_size, config.overlap),
        }

    def chunk(self, document: NormalizedDocument, artifact: dict, config: PipelineConfig, prefix: str, run_id: str) -> list[Chunk]:
        key = f"{prefix}/chunks.json"
        if not self.store.path(key).exists():
            chunks = self.strategies[config.chunker](config).chunk(document, artifact, run_id)
            if not chunks:
                raise ValueError("Document contains no extractable text")
            self.store.put(key, [chunk.model_dump() for chunk in chunks])
        return [Chunk.model_validate(chunk) for chunk in self.store.get(key)]
