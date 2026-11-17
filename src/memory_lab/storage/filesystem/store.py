import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4
from memory_lab.core.models import Document


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class FileStore:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        result = (self.root / key).resolve()
        if not result.is_relative_to(self.root.resolve()):
            raise ValueError("Invalid artifact path")
        return result

    def write(self, key: str, value: bytes) -> None:
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}")
        try:
            temporary.write_bytes(value)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def put(self, key: str, value: dict | list) -> None:
        self.write(key, json.dumps(value, ensure_ascii=False).encode())

    def get(self, key: str):
        return json.loads(self.path(key).read_text())

    @contextmanager
    def lock(self, key: str):
        path = self.path(f"locks/{key}.lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def ingest(self, filename: str, content: bytes, document_id: str | None = None) -> Document:
        content_hash = digest(content)
        # Exact upload retries share identity; explicit ID adds a content-addressed version.
        identity = document_id or content_hash
        suffix = Path(filename).suffix.lower()
        if suffix not in {".pdf", ".docx", ".md", ".txt", ".html"}:
            raise ValueError("Supported formats: PDF, DOCX, Markdown, text, HTML")
        with self.lock(identity):
            key = f"documents/{identity}/{content_hash}"
            if self.path(f"{key}/metadata.json").exists():
                return Document.model_validate(self.get(f"{key}/metadata.json"))
            doc = Document(id=identity, version=content_hash, filename=Path(filename).name, content_hash=content_hash)
            self.write(f"{key}/raw{suffix}", content)
            self.put(f"{key}/metadata.json", doc.model_dump())
            return doc

    def document(self, identity: str, version: str) -> Document:
        return Document.model_validate(self.get(f"documents/{identity}/{version}/metadata.json"))

    def raw_path(self, document: Document) -> Path:
        return self.path(f"documents/{document.id}/{document.version}/raw{Path(document.filename).suffix.lower()}")

    def documents(self) -> list[dict]:
        return [json.loads(p.read_text()) for p in sorted(self.root.glob("documents/*/*/metadata.json"))]
