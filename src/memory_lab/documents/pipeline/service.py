import json
import logging
from memory_lab.core.contracts import EmbeddingProvider, VectorIndex
from memory_lab.core.models import PipelineConfig
from memory_lab.documents.chunkers.service import ChunkingService
from memory_lab.documents.indexing.service import IndexingService
from memory_lab.documents.normalization.service import NormalizationService
from memory_lab.documents.parsers.service import ParsingService
from memory_lab.storage.filesystem.store import FileStore, digest

logger = logging.getLogger(__name__)


class DocumentPipeline:
    revision = "2"

    def __init__(self, store: FileStore, embedding: EmbeddingProvider, index: VectorIndex, embedding_identity: str, *,
        parsing: ParsingService | None = None, normalization: NormalizationService | None = None,
        chunking: ChunkingService | None = None, indexing: IndexingService | None = None):
        self.store, self.embedding, self.index = store, embedding, index
        self.namespace = digest(embedding_identity.encode())[:16]
        self.parsing = parsing or ParsingService(store)
        self.normalization = normalization or NormalizationService(store)
        self.chunking = chunking or ChunkingService(store)
        self.indexing = indexing or IndexingService(store, embedding)

    def process(self, identity: str, document_version: str, config: PipelineConfig) -> dict:
        if config.overlap >= config.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        if config.chunker == "hybrid" and config.parser != "docling":
            raise ValueError("Hybrid chunking requires Docling")
        doc = self.store.document(identity, document_version)
        parser_version = self.parsing.version(config.parser)
        spec = {**config.model_dump(), "embedding_namespace": self.namespace,
            "pipeline_revision": self.revision, "parser_version": parser_version}
        run_id = digest(json.dumps([identity, doc.version, spec], sort_keys=True).encode())
        prefix = f"runs/{run_id}"
        with self.store.lock(run_id):
            manifest = {"pipeline_run_id": run_id, "document_id": identity, "version": doc.version, "config": spec, "status": "processing"}
            self.store.put(f"{prefix}/manifest.json", manifest)
            try:
                chunks = self._artifacts(doc, config, prefix, run_id)
                self.indexing.index(chunks, prefix, self.namespace, self.index)
                manifest.update(status="completed", chunk_count=len(chunks))
            except Exception as exc:
                manifest.update(status="failed", error_type=type(exc).__name__)
                self.store.put(f"{prefix}/manifest.json", manifest)
                logger.exception("pipeline_failed", extra={"pipeline_run_id": run_id})
                raise
            self.store.put(f"{prefix}/manifest.json", manifest)
            return manifest

    def _artifacts(self, doc, config, prefix, run_id):
        parsed = self.parsing.parse(doc, config.parser, prefix)
        normalized = self.normalization.normalize(doc, parsed, prefix)
        return self.chunking.chunk(normalized, parsed.artifact, config, prefix, run_id)

    def search(self, query: str, limit: int):
        return self.index.search(self.embedding.embed([query])[0], limit, self.namespace)
