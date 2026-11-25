import json
import logging
from memory_lab.core.contracts import EmbeddingProvider, SparseIndex, VectorIndex
from memory_lab.core.models import Evidence, PipelineConfig, PipelineStageName, SearchMode
from memory_lab.documents.chunkers.service import ChunkingService
from memory_lab.documents.enrichment.service import EnrichmentService
from memory_lab.documents.indexing.service import IndexingService
from memory_lab.documents.normalization.service import NormalizationService
from memory_lab.documents.parsers.service import ParsingService
from memory_lab.retrieval.fusion import fuse_evidence
from memory_lab.retrieval.sparse import BM25Index
from memory_lab.storage.filesystem.store import FileStore, digest

logger = logging.getLogger(__name__)

STAGE_ARTIFACTS: dict[PipelineStageName, list[str]] = {
    "parse": ["parsed.json", "parsed_blocks.json", "normalized.json", "chunks.json", "enriched.json", "vectors.json"],
    "normalize": ["normalized.json", "chunks.json", "enriched.json", "vectors.json"],
    "chunk": ["chunks.json", "enriched.json", "vectors.json"],
    "enrich": ["enriched.json"],
    "index": ["vectors.json"],
}


class DocumentPipeline:
    revision = "3"

    def __init__(
        self,
        store: FileStore,
        embedding: EmbeddingProvider,
        index: VectorIndex,
        embedding_identity: str,
        *,
        parsing: ParsingService | None = None,
        normalization: NormalizationService | None = None,
        chunking: ChunkingService | None = None,
        enrichment: EnrichmentService | None = None,
        indexing: IndexingService | None = None,
        sparse_index: SparseIndex | None = None,
    ):
        self.store, self.embedding, self.index = store, embedding, index
        self.embedding_identity = embedding_identity
        self.namespace = digest(embedding_identity.encode())[:16]
        self.parsing = parsing or ParsingService(store)
        self.normalization = normalization or NormalizationService(store)
        self.chunking = chunking or ChunkingService(store)
        self.enrichment = enrichment or EnrichmentService(store)
        self.indexing = indexing or IndexingService(store, embedding)
        self.sparse_index = sparse_index or BM25Index(store)

    def process(self, identity: str, document_version: str, config: PipelineConfig) -> dict:
        if config.overlap >= config.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        if config.chunker == "hybrid" and config.parser != "docling":
            raise ValueError("Hybrid chunking requires Docling")
        doc = self.store.document(identity, document_version)
        parser_version = self.parsing.version(config.parser)
        spec = {
            **config.model_dump(),
            "embedding_namespace": self.namespace,
            "pipeline_revision": self.revision,
            "parser_version": parser_version,
        }
        run_id = digest(json.dumps([identity, doc.version, spec], sort_keys=True).encode())
        prefix = f"runs/{run_id}"
        with self.store.lock(run_id):
            manifest = {
                "pipeline_run_id": run_id,
                "document_id": identity,
                "version": doc.version,
                "config": spec,
                "status": "processing",
            }
            self.store.put(f"{prefix}/manifest.json", manifest)
            try:
                chunks, enrichment_summary = self._artifacts(doc, config, prefix, run_id)
                self.indexing.index(chunks, prefix, self.namespace, self.index)
                self.sparse_index.upsert(chunks, self.namespace)
                manifest.update(
                    status="completed",
                    chunk_count=len(chunks),
                    stages={
                        "parse": "completed",
                        "normalize": "completed",
                        "chunk": "completed",
                        "enrich": "completed" if config.enrich else "skipped",
                        "index": "completed",
                    },
                    enrichment_summary=enrichment_summary,
                )
            except Exception as exc:
                manifest.update(status="failed", error_type=type(exc).__name__)
                self.store.put(f"{prefix}/manifest.json", manifest)
                logger.exception("pipeline_failed", extra={"pipeline_run_id": run_id})
                raise
            self.store.put(f"{prefix}/manifest.json", manifest)
            return manifest

    def _artifacts(self, doc, config: PipelineConfig, prefix: str, run_id: str):
        parsed = self.parsing.parse(doc, config.parser, prefix)
        normalized = self.normalization.normalize(doc, parsed, prefix)
        chunks = self.chunking.chunk(normalized, parsed.artifact, config, prefix, run_id)
        enrichment_summary = None
        if config.enrich:
            enriched = self.enrichment.enrich(
                chunks, prefix=prefix, pipeline_run_id=run_id, document_id=doc.id, version=doc.version
            )
            enrichment_summary = {
                "entity_count": len(enriched.entities),
                "relation_count": len(enriched.relations),
                "temporal_count": len(enriched.temporal),
            }
        return chunks, enrichment_summary

    def invalidate(self, run_id: str, stages: list[PipelineStageName]) -> dict:
        prefix = f"runs/{run_id}"
        manifest_key = f"{prefix}/manifest.json"
        with self.store.lock(run_id):
            manifest = self.store.get(manifest_key)
            removed: list[str] = []
            files_to_remove: set[str] = set()
            for stage in stages:
                if stage not in STAGE_ARTIFACTS:
                    raise ValueError(f"Unknown pipeline stage: {stage}")
                files_to_remove.update(STAGE_ARTIFACTS[stage])
            for filename in sorted(files_to_remove):
                artifact_path = self.store.path(f"{prefix}/{filename}")
                if artifact_path.exists():
                    artifact_path.unlink()
                    removed.append(filename)
            stage_status = dict(manifest.get("stages") or {})
            for stage, affected in STAGE_ARTIFACTS.items():
                if any(f in files_to_remove for f in affected[:1]) or stage in stages:
                    stage_status[stage] = "invalidated"
            manifest["status"] = "invalidated"
            manifest["stages"] = stage_status
            manifest["invalidated_stages"] = sorted(set(stages))
            manifest["removed_artifacts"] = removed
            self.store.put(manifest_key, manifest)
            return manifest

    def search(
        self,
        query: str,
        limit: int,
        *,
        document_id: str | None = None,
        version: str | None = None,
        pipeline_run_id: str | None = None,
        mode: SearchMode = "dense",
    ) -> list[Evidence]:
        filters = {
            k: v
            for k, v in {
                "document_id": document_id,
                "version": version,
                "pipeline_run_id": pipeline_run_id,
            }.items()
            if v
        } or None

        if mode == "sparse":
            return self.sparse_index.search(query, limit, self.namespace, filters=filters)

        if mode == "dense":
            vector = self.embedding.embed([query])[0]
            return self._dense_search(vector, limit, filters)

        if mode == "hybrid":
            candidate_limit = max(limit * 3, 15)
            sparse_hits = self.sparse_index.search(query, candidate_limit, self.namespace, filters=filters)
            dense_hits: list[Evidence] = []
            try:
                vector = self.embedding.embed([query])[0]
                dense_hits = self._dense_search(vector, candidate_limit, filters)
            except Exception:
                logger.warning("hybrid_search_dense_unavailable_falling_back_to_sparse")
            return fuse_evidence(dense_hits, sparse_hits, limit)

        raise ValueError(f"Unsupported search mode: {mode}")

    def _dense_search(
        self, vector: list[float], limit: int, filters: dict[str, str] | None
    ) -> list[Evidence]:
        if filters:
            return self.index.search(vector, limit, self.namespace, filters=filters)
        return self.index.search(vector, limit, self.namespace)
