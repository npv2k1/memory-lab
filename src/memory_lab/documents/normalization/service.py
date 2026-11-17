from memory_lab.core.models import Document, NormalizedDocument, ParsedDocument
from memory_lab.storage.filesystem.store import FileStore


class NormalizationService:
    """Bind parser-independent blocks to their immutable document version."""

    def __init__(self, store: FileStore):
        self.store = store

    def normalize(self, document: Document, parsed: ParsedDocument, prefix: str) -> NormalizedDocument:
        key = f"{prefix}/normalized.json"
        if self.store.path(key).exists():
            return NormalizedDocument.model_validate(self.store.get(key))
        block_ids = [block.id for block in parsed.blocks]
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("Parser returned duplicate block IDs")
        result = NormalizedDocument(document_id=document.id, version=document.version, blocks=parsed.blocks)
        self.store.put(key, result.model_dump())
        return result
