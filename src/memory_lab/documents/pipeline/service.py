import json
import logging
import shutil
from memory_lab.core.contracts import EmbeddingProvider, SparseIndex, VectorIndex
from memory_lab.core.models import Evidence, PipelineConfig, PipelineStageName, SearchMode, utc_now_iso
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

    def configure_embedding(self, embedding: EmbeddingProvider, embedding_identity: str) -> None:
        """Hot-swap the active embedding provider and namespace without restarting."""
        self.embedding = embedding
        self.embedding_identity = embedding_identity
        self.namespace = digest(embedding_identity.encode())[:16]
        self.indexing.embedding = embedding

    def _validate_and_spec(self, identity: str, document_version: str, config: PipelineConfig):
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
        return doc, spec, run_id

    def prepare_run(self, identity: str, document_version: str, config: PipelineConfig) -> dict:
        """Validate config and write initial manifest for background processing."""
        doc, spec, run_id = self._validate_and_spec(identity, document_version, config)
        prefix = f"runs/{run_id}"
        now = utc_now_iso()
        manifest = {
            "pipeline_run_id": run_id,
            "document_id": identity,
            "version": doc.version,
            "config": spec,
            "status": "processing",
            "current_stage": "queued",
            "progress": 5,
            "stages": {
                "parse": "pending",
                "normalize": "pending",
                "chunk": "pending",
                "enrich": "pending" if config.enrich else "skipped",
                "index": "pending",
            },
            "stage_details": {},
            "started_at": now,
            "updated_at": now,
        }
        with self.store.lock(run_id):
            self.store.put(f"{prefix}/manifest.json", manifest)
        return manifest

    def _persist_progress(
        self,
        prefix: str,
        manifest: dict,
        *,
        stage: str,
        progress: int,
        stage_status: str | None = None,
        detail: dict | None = None,
    ) -> None:
        manifest["current_stage"] = stage
        manifest["progress"] = progress
        if stage_status and stage in manifest["stages"]:
            manifest["stages"][stage] = stage_status
        if detail:
            manifest.setdefault("stage_details", {})[stage] = detail
        manifest["updated_at"] = utc_now_iso()
        self.store.put(f"{prefix}/manifest.json", manifest)

    def process(self, identity: str, document_version: str, config: PipelineConfig) -> dict:
        doc, spec, run_id = self._validate_and_spec(identity, document_version, config)
        prefix = f"runs/{run_id}"
        now = utc_now_iso()
        with self.store.lock(run_id):
            manifest = {
                "pipeline_run_id": run_id,
                "document_id": identity,
                "version": doc.version,
                "config": spec,
                "status": "processing",
                "current_stage": "parse",
                "progress": 10,
                "stages": {
                    "parse": "running",
                    "normalize": "pending",
                    "chunk": "pending",
                    "enrich": "pending" if config.enrich else "skipped",
                    "index": "pending",
                },
                "stage_details": {},
                "started_at": now,
                "updated_at": now,
            }
            self.store.put(f"{prefix}/manifest.json", manifest)
            try:
                chunks, enrichment_summary = self._artifacts(doc, config, prefix, run_id, manifest)
                self._persist_progress(prefix, manifest, stage="index", progress=88, stage_status="running")
                self.indexing.index(chunks, prefix, self.namespace, self.index)
                self.sparse_index.upsert(chunks, self.namespace)
                completed_at = utc_now_iso()
                manifest.update(
                    status="completed",
                    current_stage="completed",
                    progress=100,
                    chunk_count=len(chunks),
                    stages={
                        "parse": "completed",
                        "normalize": "completed",
                        "chunk": "completed",
                        "enrich": "completed" if config.enrich else "skipped",
                        "index": "completed",
                    },
                    enrichment_summary=enrichment_summary,
                    updated_at=completed_at,
                    completed_at=completed_at,
                )
            except Exception as exc:
                failed_stage = manifest.get("current_stage", "unknown")
                if failed_stage in manifest.get("stages", {}):
                    manifest["stages"][failed_stage] = "failed"
                manifest.update(
                    status="failed",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    updated_at=utc_now_iso(),
                )
                self.store.put(f"{prefix}/manifest.json", manifest)
                logger.exception("pipeline_failed", extra={"pipeline_run_id": run_id})
                raise
            self.store.put(f"{prefix}/manifest.json", manifest)
            return manifest

    def _artifacts(self, doc, config: PipelineConfig, prefix: str, run_id: str, manifest: dict | None = None):
        parsed = self.parsing.parse(doc, config.parser, prefix)
        if manifest is not None:
            self._persist_progress(
                prefix, manifest, stage="parse", progress=25, stage_status="completed",
                detail={"block_count": len(parsed.blocks)},
            )
            self._persist_progress(prefix, manifest, stage="normalize", progress=35, stage_status="running")

        normalized = self.normalization.normalize(doc, parsed, prefix)
        if manifest is not None:
            self._persist_progress(
                prefix, manifest, stage="normalize", progress=45, stage_status="completed",
                detail={"block_count": len(normalized.blocks)},
            )
            self._persist_progress(prefix, manifest, stage="chunk", progress=55, stage_status="running")

        chunks = self.chunking.chunk(normalized, parsed.artifact, config, prefix, run_id)
        if manifest is not None:
            self._persist_progress(
                prefix, manifest, stage="chunk", progress=68, stage_status="completed",
                detail={"chunk_count": len(chunks)},
            )

        enrichment_summary = None
        if config.enrich:
            if manifest is not None:
                self._persist_progress(prefix, manifest, stage="enrich", progress=75, stage_status="running")
            enriched = self.enrichment.enrich(
                chunks, prefix=prefix, pipeline_run_id=run_id, document_id=doc.id, version=doc.version
            )
            enrichment_summary = {
                "entity_count": len(enriched.entities),
                "relation_count": len(enriched.relations),
                "temporal_count": len(enriched.temporal),
            }
            if manifest is not None:
                self._persist_progress(
                    prefix, manifest, stage="enrich", progress=84, stage_status="completed",
                    detail=enrichment_summary,
                )
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
            manifest["updated_at"] = utc_now_iso()
            self.store.put(manifest_key, manifest)
            return manifest

    def delete_run(self, run_id: str) -> dict:
        """Delete a pipeline run's artifacts and remove its indexed chunks from Qdrant and BM25."""
        run_dir = self.store.path(f"runs/{run_id}")
        if not run_dir.exists():
            raise FileNotFoundError(f"Pipeline run not found: {run_id}")
        namespaces = {self.namespace}
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                ns = (manifest.get("config") or {}).get("embedding_namespace")
                if ns:
                    namespaces.add(ns)
            except Exception:
                pass
        with self.store.lock(run_id):
            for ns in namespaces:
                if hasattr(self.index, "delete_by_filter"):
                    try:
                        self.index.delete_by_filter({"pipeline_run_id": run_id}, ns)
                    except Exception:
                        logger.warning("qdrant_delete_run_failed", extra={"pipeline_run_id": run_id})
                if hasattr(self.sparse_index, "delete_by_filter"):
                    try:
                        self.sparse_index.delete_by_filter({"pipeline_run_id": run_id}, ns)
                    except Exception:
                        logger.warning("bm25_delete_run_failed", extra={"pipeline_run_id": run_id})
            shutil.rmtree(run_dir, ignore_errors=True)
        return {"deleted": True, "pipeline_run_id": run_id}

    def delete_document(self, identity: str, version: str | None = None) -> dict:
        """Delete a document (or specific version), its runs, and all indexed Qdrant/BM25 entries."""
        doc_root = self.store.path(f"documents/{identity}")
        if not doc_root.exists():
            raise FileNotFoundError(f"Document not found: {identity}")
        if version:
            target_dir = doc_root / version
            if not target_dir.exists():
                raise FileNotFoundError(f"Document version not found: {identity}/{version}")
            versions_to_delete = [version]
        else:
            versions_to_delete = sorted(p.name for p in doc_root.iterdir() if p.is_dir())

        deleted_runs: list[str] = []
        namespaces = {self.namespace}
        runs_root = self.store.path("runs")
        if runs_root.exists():
            for manifest_path in list(runs_root.glob("*/manifest.json")):
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if manifest.get("document_id") != identity:
                    continue
                if version and manifest.get("version") != version:
                    continue
                run_id = manifest.get("pipeline_run_id") or manifest_path.parent.name
                ns = (manifest.get("config") or {}).get("embedding_namespace")
                if ns:
                    namespaces.add(ns)
                try:
                    self.delete_run(run_id)
                    deleted_runs.append(run_id)
                except FileNotFoundError:
                    pass

        # Ensure any orphaned index entries for this document/version are also removed
        index_filter = {"document_id": identity}
        if version:
            index_filter["version"] = version
        for ns in namespaces:
            if hasattr(self.index, "delete_by_filter"):
                try:
                    self.index.delete_by_filter(index_filter, ns)
                except Exception:
                    logger.warning("qdrant_delete_document_failed", extra={"document_id": identity})
            if hasattr(self.sparse_index, "delete_by_filter"):
                try:
                    self.sparse_index.delete_by_filter(index_filter, ns)
                except Exception:
                    logger.warning("bm25_delete_document_failed", extra={"document_id": identity})

        for ver in versions_to_delete:
            shutil.rmtree(doc_root / ver, ignore_errors=True)
        # Clean up empty parent document folder
        if doc_root.exists() and not any(doc_root.iterdir()):
            shutil.rmtree(doc_root, ignore_errors=True)

        return {
            "deleted": True,
            "document_id": identity,
            "deleted_versions": versions_to_delete,
            "deleted_runs": deleted_runs,
        }

    def _latest_completed_runs(self) -> dict[tuple[str, str], str]:
        """Map (document_id, version) -> most recent completed pipeline_run_id in active namespace."""
        latest: dict[tuple[str, str], tuple[str, float, str]] = {}
        runs_root = self.store.path("runs")
        if not runs_root.exists():
            return {}
        for manifest_path in runs_root.glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if manifest.get("status") != "completed":
                continue
            ns = (manifest.get("config") or {}).get("embedding_namespace")
            if ns and ns != self.namespace:
                continue
            doc_id = manifest.get("document_id")
            ver = manifest.get("version")
            run_id = manifest.get("pipeline_run_id") or manifest_path.parent.name
            if not doc_id or not ver:
                continue
            ts = str(manifest.get("completed_at") or manifest.get("updated_at") or "")
            mtime = manifest_path.stat().st_mtime
            key = (doc_id, ver)
            prev = latest.get(key)
            if prev is None or (ts, mtime, run_id) > prev:
                latest[key] = (ts, mtime, run_id)
        return {k: v[2] for k, v in latest.items()}

    def _dedupe_hits(self, hits: list[Evidence], limit: int) -> list[Evidence]:
        if not hits:
            return []
        latest_runs = self._latest_completed_runs()
        deduped: list[Evidence] = []
        seen_signatures: set[tuple[str, str, str]] = set()
        for hit in hits:
            doc_ver = (hit.chunk.document_id, hit.chunk.version)
            preferred_run = latest_runs.get(doc_ver)
            if preferred_run and hit.chunk.pipeline_run_id != preferred_run:
                continue
            sig = (hit.chunk.document_id, hit.chunk.version, " ".join(hit.chunk.text.split()).lower())
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            deduped.append(hit)
            if len(deduped) >= limit:
                break
        return deduped

    def search(
        self,
        query: str,
        limit: int,
        *,
        document_id: str | None = None,
        version: str | None = None,
        pipeline_run_id: str | None = None,
        mode: SearchMode = "dense",
        dedupe: bool = True,
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

        should_dedupe = dedupe and pipeline_run_id is None
        fetch_limit = max(limit * 4, 20) if should_dedupe else limit

        if mode == "sparse":
            raw = self.sparse_index.search(query, fetch_limit, self.namespace, filters=filters)
            return self._dedupe_hits(raw, limit) if should_dedupe else raw[:limit]

        if mode == "dense":
            vector = self.embedding.embed([query])[0]
            raw = self._dense_search(vector, fetch_limit, filters)
            return self._dedupe_hits(raw, limit) if should_dedupe else raw[:limit]

        if mode == "hybrid":
            candidate_limit = max(fetch_limit * 2, 20)
            sparse_hits = self.sparse_index.search(query, candidate_limit, self.namespace, filters=filters)
            dense_hits: list[Evidence] = []
            try:
                vector = self.embedding.embed([query])[0]
                dense_hits = self._dense_search(vector, candidate_limit, filters)
            except Exception:
                logger.warning("hybrid_search_dense_unavailable_falling_back_to_sparse")
            fused = fuse_evidence(dense_hits, sparse_hits, fetch_limit)
            return self._dedupe_hits(fused, limit) if should_dedupe else fused[:limit]

        raise ValueError(f"Unsupported search mode: {mode}")

    def _dense_search(
        self, vector: list[float], limit: int, filters: dict[str, str] | None
    ) -> list[Evidence]:
        if filters:
            return self.index.search(vector, limit, self.namespace, filters=filters)
        return self.index.search(vector, limit, self.namespace)
