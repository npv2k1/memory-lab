from memory_lab.core.contracts import DocumentParser
from memory_lab.core.models import Document, ParsedDocument
from memory_lab.documents.parsers.adapters import DoclingParser, TextParser
from memory_lab.storage.filesystem.store import FileStore


class ParsingService:
    def __init__(self, store: FileStore, parsers: dict[str, DocumentParser] | None = None):
        self.store = store
        self.parsers = parsers if parsers is not None else {"text": TextParser(), "docling": DoclingParser()}

    def version(self, name: str) -> str:
        return self.parsers[name].version

    def parse(self, document: Document, parser_name: str, prefix: str) -> ParsedDocument:
        key = f"{prefix}/parsed_blocks.json"
        if self.store.path(key).exists():
            return ParsedDocument.model_validate(self.store.get(key))
        result = self.parsers[parser_name].parse(self.store.raw_path(document))
        self.store.put(f"{prefix}/parsed.json", result.artifact)
        self.store.put(key, result.model_dump())
        return result
