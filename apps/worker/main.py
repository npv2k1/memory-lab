"""Horizontally scalable background worker for async document processing & indexing jobs."""
import argparse
import logging
import os
import signal
import socket
import time
from memory_lab.core.config import Settings
from memory_lab.documents.ingestion.service import IngestionService
from memory_lab.documents.pipeline.service import DocumentPipeline
from memory_lab.enterprise.jobs import JobQueue
from memory_lab.enterprise.store import EnterpriseStore
from memory_lab.enterprise.tenant_registry import TenantContextRegistry, TenantServices
from memory_lab.experiments.service import ExperimentService
from memory_lab.knowledge.atlas import EmbeddingAtlasService
from memory_lab.knowledge.service import KnowledgeService
from memory_lab.mcp.server import MCPServer
from memory_lab.memory.service import MemoryService
from memory_lab.retrieval.embedding import OpenAIEmbeddings
from memory_lab.storage.filesystem.store import FileStore
from memory_lab.storage.graph.store import FileGraphStore, Neo4jGraphStore
from memory_lab.storage.qdrant.index import QdrantIndex, create_qdrant_client

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
)
logger = logging.getLogger("memory_lab.worker")


def build_worker_runtime(settings: Settings | None = None) -> tuple[JobQueue, TenantContextRegistry]:
    settings = settings or Settings()
    store = FileStore(settings.data_dir)
    embedding = OpenAIEmbeddings(settings.embedding_base_url, settings.embedding_api_key, settings.embedding_model)
    index = QdrantIndex(create_qdrant_client(settings.qdrant_url), settings.qdrant_collection)
    emb_id = f"{settings.embedding_base_url}:{settings.embedding_model}"
    pipeline = DocumentPipeline(store, embedding, index, emb_id)

    file_graph = FileGraphStore(store)
    graph = (
        Neo4jGraphStore(
            settings.neo4j_url, settings.neo4j_user, settings.neo4j_password, settings.neo4j_database, mirror=file_graph
        )
        if settings.neo4j_url
        else file_graph
    )
    memory = MemoryService(
        store, embedding, index, emb_id, graph=graph, default_strategy=settings.default_memory_strategy  # type: ignore[arg-type]
    )
    default_services = TenantServices(
        tenant_id=settings.default_tenant_id,
        store=store,
        ingestion=IngestionService(store),
        pipeline=pipeline,
        memory=memory,
        experiments=ExperimentService(store, pipeline, memory),
        knowledge=KnowledgeService(store, pipeline, memory),
        atlas=EmbeddingAtlasService(store, memory),
        mcp=MCPServer(memory),
    )
    registry = TenantContextRegistry(settings, default_services)
    ent_store = EnterpriseStore(settings.data_dir, settings.database_url)
    queue = JobQueue(ent_store, redis_url=settings.redis_url)
    return queue, registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory Lab Enterprise Async Worker")
    parser.add_argument("--once", action="store_true", help="Process at most one queued job and exit")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Idle poll interval in seconds")
    args = parser.parse_args()

    worker_id = f"{socket.gethostname()}-{os.getpid()}"
    queue, registry = build_worker_runtime()
    running = True

    def _stop(_signum, _frame):
        nonlocal running
        logger.info("worker_shutting_down")
        running = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    logger.info("worker_started worker_id=%s backend=%s", worker_id, queue.store.backend_name)
    while running:
        job = queue.pop_and_execute(registry, worker_id=worker_id, block_seconds=2)
        if job:
            logger.info("worker_finished_job job_id=%s status=%s", job["id"], job["status"])
            if args.once:
                break
        else:
            if args.once:
                break
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
