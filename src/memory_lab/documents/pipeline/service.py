import json
import logging
from importlib.metadata import version
from memory_lab.core.contracts import EmbeddingProvider, VectorIndex
from memory_lab.core.models import Chunk, NormalizedDocument, PipelineConfig
from memory_lab.documents.chunkers.strategies import FixedChunker, HybridChunkerAdapter
from memory_lab.documents.parsers.adapters import DoclingParser, TextParser
from memory_lab.storage.filesystem.store import FileStore, digest

logger = logging.getLogger(__name__)


class DocumentPipeline:
    revision = "1"

    def __init__(self, store: FileStore, embedding: EmbeddingProvider, index: VectorIndex, embedding_identity: str):
        self.store, self.embedding, self.index = store, embedding, index
        self.namespace = digest(embedding_identity.encode())[:16]

    def process(self, identity: str, document_version: str, config: PipelineConfig) -> dict:
        if config.overlap >= config.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        if config.chunker == "hybrid" and config.parser != "docling":
            raise ValueError("Hybrid chunking requires Docling")
        doc = self.store.document(identity, document_version)
        parser_version = version("docling") if config.parser == "docling" else "1"
        spec = {**config.model_dump(), "embedding_namespace": self.namespace,
            "pipeline_revision": self.revision, "parser_version": parser_version}
        run_id = digest(json.dumps([identity, doc.version, spec], sort_keys=True).encode())
        prefix = f"runs/{run_id}"
        with self.store.lock(run_id):
            manifest = {"pipeline_run_id": run_id, "document_id": identity, "version": doc.version, "config": spec, "status": "processing"}
            self.store.put(f"{prefix}/manifest.json", manifest)
            try:
                chunks = self._artifacts(doc, config, prefix, run_id)
                # Re-upsert deterministic IDs, including after Qdrant has been rebuilt.
                if not self.store.path(f"{prefix}/vectors.json").exists():
                    self.store.put(f"{prefix}/vectors.json", self.embedding.embed([c.text for c in chunks]))
                self.index.upsert(chunks, self.store.get(f"{prefix}/vectors.json"), self.namespace)
                manifest.update(status="completed", chunk_count=len(chunks))
            except Exception as exc:
                manifest.update(status="failed", error_type=type(exc).__name__)
                self.store.put(f"{prefix}/manifest.json", manifest)
                logger.exception("pipeline_failed", extra={"pipeline_run_id": run_id})
                raise
            self.store.put(f"{prefix}/manifest.json", manifest)
            return manifest

    def _artifacts(self, doc, config, prefix, run_id):
        if not self.store.path(f"{prefix}/normalized.json").exists():
            parser = DoclingParser() if config.parser == "docling" else TextParser()
            normalized, artifact = parser.parse(self.store.raw_path(doc), doc)
            self.store.put(f"{prefix}/parsed.json", artifact)
            self.store.put(f"{prefix}/normalized.json", normalized.model_dump())
        normalized = NormalizedDocument.model_validate(self.store.get(f"{prefix}/normalized.json"))
        if not self.store.path(f"{prefix}/chunks.json").exists():
            chunker = HybridChunkerAdapter() if config.chunker == "hybrid" else FixedChunker(config.chunk_size, config.overlap)
            chunks = chunker.chunk(normalized, self.store.get(f"{prefix}/parsed.json"), run_id)
            if not chunks:
                raise ValueError("Document contains no extractable text")
            self.store.put(f"{prefix}/chunks.json", [chunk.model_dump() for chunk in chunks])
        return [Chunk.model_validate(c) for c in self.store.get(f"{prefix}/chunks.json")]

    def search(self, query: str, limit: int):
        return self.index.search(self.embedding.embed([query])[0], limit, self.namespace)
