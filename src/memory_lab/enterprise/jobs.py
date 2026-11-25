import logging
import socket
from typing import Any
from memory_lab.core.models import PipelineConfig
from memory_lab.enterprise.store import EnterpriseStore
from memory_lab.enterprise.tenant_registry import TenantContextRegistry

logger = logging.getLogger(__name__)
REDIS_QUEUE_KEY = "mlab:jobs:queue"


class JobQueue:
    """Distributed async job queue backed by EnterpriseStore (Postgres/SQLite) + optional Redis wake-up list."""

    def __init__(self, store: EnterpriseStore, *, redis_url: str = ""):
        self.store = store
        self._redis: Any = None
        if redis_url.strip():
            try:
                import redis  # type: ignore[import-not-found]
                self._redis = redis.Redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=5)
            except Exception:
                self._redis = None

    def enqueue_document_process(
        self,
        *,
        tenant_id: str,
        document_id: str,
        version: str,
        config: PipelineConfig,
    ) -> dict[str, Any]:
        job = self.store.create_job(
            tenant_id,
            "document_process",
            {"document_id": document_id, "version": version, "config": config.model_dump()},
        )
        if self._redis is not None:
            try:
                self._redis.lpush(REDIS_QUEUE_KEY, job["id"])
            except Exception:
                logger.warning("redis_job_push_failed", extra={"job_id": job["id"]})
        return job

    def execute_job(
        self,
        job: dict[str, Any],
        registry: TenantContextRegistry,
        *,
        worker_id: str | None = None,
    ) -> dict[str, Any]:
        wid = worker_id or socket.gethostname()
        job_id = job["id"]
        self.store.update_job(job_id, status="running", progress=25, worker_id=wid)
        try:
            tenant_services = registry.get(job["tenant_id"])
            if job["job_type"] == "document_process":
                payload = job["payload"]
                config = PipelineConfig.model_validate(payload["config"])
                manifest = tenant_services.pipeline.process(
                    payload["document_id"], payload["version"], config
                )
                return self.store.update_job(
                    job_id, status="completed", progress=100, result=manifest, worker_id=wid
                ) or job
            raise ValueError(f"Unsupported job_type: {job['job_type']}")
        except Exception as exc:
            logger.exception("worker_job_failed", extra={"job_id": job_id, "tenant_id": job["tenant_id"]})
            return self.store.update_job(
                job_id,
                status="failed",
                progress=100,
                error=f"{type(exc).__name__}: {exc}",
                worker_id=wid,
            ) or job

    def pop_and_execute(
        self,
        registry: TenantContextRegistry,
        *,
        worker_id: str | None = None,
        block_seconds: int = 2,
    ) -> dict[str, Any] | None:
        wid = worker_id or socket.gethostname()
        if self._redis is not None:
            try:
                popped = self._redis.brpop(REDIS_QUEUE_KEY, timeout=block_seconds)
                if popped:
                    raw_id = popped[1]
                    job_id = raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)
                    job = self.store.get_job(job_id)
                    if job and job["status"] == "queued":
                        return self.execute_job(job, registry, worker_id=wid)
            except Exception:
                pass

        claimed = self.store.claim_next_queued_job(wid)
        if not claimed:
            return None
        return self.execute_job(claimed, registry, worker_id=wid)
