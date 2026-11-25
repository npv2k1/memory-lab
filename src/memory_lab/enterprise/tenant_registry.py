from collections import OrderedDict
from dataclasses import dataclass
import threading
from typing import Any
from memory_lab.core.config import Settings
from memory_lab.documents.ingestion.service import IngestionService
from memory_lab.documents.pipeline.service import DocumentPipeline
from memory_lab.experiments.service import ExperimentService
from memory_lab.knowledge.atlas import EmbeddingAtlasService
from memory_lab.knowledge.service import KnowledgeService
from memory_lab.mcp.server import MCPServer
from memory_lab.memory.service import MemoryService
from memory_lab.storage.filesystem.store import FileStore
from memory_lab.storage.graph.store import FileGraphStore, Neo4jGraphStore
from memory_lab.storage.qdrant.index import QdrantIndex


@dataclass
class TenantServices:
    tenant_id: str
    store: FileStore
    ingestion: IngestionService
    pipeline: DocumentPipeline
    memory: MemoryService
    experiments: ExperimentService
    knowledge: KnowledgeService
    atlas: EmbeddingAtlasService
    mcp: MCPServer


class TenantContextRegistry:
    """Manages physically and logically isolated per-tenant service graphs with bounded LRU caching."""

    def __init__(
        self,
        settings: Settings,
        default_services: TenantServices,
        *,
        max_cached_tenants: int = 128,
    ):
        self.settings = settings
        self.default_services = default_services
        self.max_cached_tenants = max_cached_tenants
        self._lock = threading.RLock()
        self._cache: OrderedDict[str, TenantServices] = OrderedDict()

    def get(self, tenant_id: str) -> TenantServices:
        safe_id = tenant_id.strip().lower() or self.settings.default_tenant_id
        if safe_id == self.settings.default_tenant_id:
            return self.default_services

        with self._lock:
            existing = self._cache.get(safe_id)
            if existing is not None:
                self._cache.move_to_end(safe_id)
                return existing

            tenant_root = self.default_services.store.root / "tenants" / safe_id
            store = FileStore(tenant_root)
            ingestion = IngestionService(store)

            base_index: Any = self.default_services.pipeline.index
            if isinstance(base_index, QdrantIndex):
                tenant_index: Any = QdrantIndex(base_index.client, f"{base_index.collection}_t_{safe_id}")
            else:
                tenant_index = base_index

            emb = self.default_services.pipeline.embedding
            emb_id = f"{safe_id}:{self.default_services.pipeline.embedding_identity}"
            pipeline = DocumentPipeline(store, emb, tenant_index, emb_id)

            file_graph = FileGraphStore(store)
            graph = (
                Neo4jGraphStore(
                    self.settings.neo4j_url,
                    self.settings.neo4j_user,
                    self.settings.neo4j_password,
                    self.settings.neo4j_database,
                    mirror=file_graph,
                )
                if self.settings.neo4j_url
                else file_graph
            )
            memory = MemoryService(
                store,
                emb,
                tenant_index,
                emb_id,
                graph=graph,
                default_strategy=self.settings.default_memory_strategy,  # type: ignore[arg-type]
            )
            experiments = ExperimentService(store, pipeline, memory)
            knowledge = KnowledgeService(store, pipeline, memory)
            atlas = EmbeddingAtlasService(store, memory)
            mcp = MCPServer(memory)

            services = TenantServices(
                tenant_id=safe_id,
                store=store,
                ingestion=ingestion,
                pipeline=pipeline,
                memory=memory,
                experiments=experiments,
                knowledge=knowledge,
                atlas=atlas,
                mcp=mcp,
            )
            self._cache[safe_id] = services
            while len(self._cache) > self.max_cached_tenants:
                self._cache.popitem(last=False)
            return services
