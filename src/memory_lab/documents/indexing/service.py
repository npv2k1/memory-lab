from memory_lab.core.contracts import EmbeddingProvider, VectorIndex
from memory_lab.core.models import Chunk
from memory_lab.storage.filesystem.store import FileStore


class IndexingService:
    def __init__(self, store: FileStore, embedding: EmbeddingProvider):
        self.store, self.embedding = store, embedding

    def index(self, chunks: list[Chunk], prefix: str, namespace: str, index: VectorIndex) -> None:
        key = f"{prefix}/vectors.json"
        if not self.store.path(key).exists():
            self.store.put(key, self.embedding.embed([chunk.text for chunk in chunks]))
        # Upsert even on retry: indexes can be rebuilt independently from artifacts.
        index.upsert(chunks, self.store.get(key), namespace)
