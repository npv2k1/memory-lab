from memory_lab.core.contracts import EnrichmentExtractor
from memory_lab.core.models import Chunk, EnrichmentArtifact
from memory_lab.documents.enrichment.extractor import NonGenerativeExtractor
from memory_lab.storage.filesystem.store import FileStore


class EnrichmentService:
    """Non-generative enrichment stage that caches entity, relation, and temporal metadata per run."""

    def __init__(self, store: FileStore, extractor: EnrichmentExtractor | None = None):
        self.store = store
        self.extractor = extractor or NonGenerativeExtractor()

    def enrich(
        self, chunks: list[Chunk], *, prefix: str, pipeline_run_id: str, document_id: str, version: str
    ) -> EnrichmentArtifact:
        key = f"{prefix}/enriched.json"
        if not self.store.path(key).exists():
            artifact = self.extractor.extract_chunks(
                chunks, pipeline_run_id=pipeline_run_id, document_id=document_id, version=version
            )
            self.store.put(key, artifact.model_dump())
        return EnrichmentArtifact.model_validate(self.store.get(key))
