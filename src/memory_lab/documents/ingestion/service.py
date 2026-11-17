from memory_lab.core.models import Document
from memory_lab.storage.filesystem.store import FileStore


class IngestionService:
    """Accept document bytes; persistence owns hashing, versions and deduplication."""

    def __init__(self, store: FileStore):
        self.store = store

    def ingest(self, filename: str, content: bytes, document_id: str | None = None) -> Document:
        if not content:
            raise ValueError("File is empty")
        return self.store.ingest(filename, content, document_id)
