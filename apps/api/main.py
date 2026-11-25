import asyncio
import hmac
import json
import time
from typing import Any
from uuid import UUID, uuid4
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from memory_lab.core.config import Settings
from memory_lab.core.models import (
    BenchmarkDatasetCreate,
    ExperimentRunRequest,
    MemorySearchRequest,
    MemoryStoreRequest,
    MemoryStrategyName,
    PipelineConfig,
    SearchMode,
    StageInvalidationRequest,
)
from memory_lab.documents.ingestion.service import IngestionService
from memory_lab.documents.pipeline.service import DocumentPipeline
from memory_lab.enterprise.auth import AuthContext, AuthService, Role
from memory_lab.enterprise.embedding_settings import (
    EmbeddingSettingsService,
    EmbeddingSettingsUpdate,
    EmbeddingTestRequest,
)
from memory_lab.enterprise.jobs import JobQueue
from memory_lab.enterprise.metrics import MetricsRegistry
from memory_lab.enterprise.object_store import S3ObjectMirror
from memory_lab.enterprise.ratelimit import RateLimiter
from memory_lab.enterprise.store import EnterpriseStore
from memory_lab.enterprise.tenant_registry import TenantContextRegistry, TenantServices
from memory_lab.experiments.service import ExperimentService
from memory_lab.knowledge.atlas import AtlasSourceFilter, EmbeddingAtlasService
from memory_lab.knowledge.service import KnowledgeService, KnowledgeSourceFilter, TableViewName
from memory_lab.mcp.server import MCPServer
from memory_lab.memory.service import MemoryService
from memory_lab.retrieval.embedding import OpenAIEmbeddings
from memory_lab.storage.filesystem.store import FileStore
from memory_lab.storage.graph.store import FileGraphStore, Neo4jGraphStore
from memory_lab.storage.qdrant.index import QdrantIndex, create_qdrant_client


class TenantCreateRequest(BaseModel):
    id: str = Field(min_length=2, max_length=64, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    name: str = Field(min_length=1, max_length=120)
    plan: str = "enterprise"
    max_documents: int = Field(default=10000, ge=1)
    max_memories: int = Field(default=100000, ge=1)
    rate_limit_rpm: int = Field(default=600, ge=0)


class ApiKeyCreateRequest(BaseModel):
    tenant_id: str | None = None
    name: str = Field(min_length=1, max_length=120)
    role: Role = "editor"


class AsyncProcessJobRequest(BaseModel):
    document_id: str
    version: str
    config: PipelineConfig = Field(default_factory=PipelineConfig)


def create_app(
    settings: Settings | None = None,
    pipeline: DocumentPipeline | None = None,
    memory: MemoryService | None = None,
    experiments: ExperimentService | None = None,
) -> FastAPI:
    settings = settings or Settings()
    store = pipeline.store if pipeline else FileStore(settings.data_dir)
    injected_pipeline = pipeline is not None
    if pipeline is None:
        embedding = OpenAIEmbeddings(settings.embedding_base_url, settings.embedding_api_key, settings.embedding_model)
        index = QdrantIndex(create_qdrant_client(settings.qdrant_url), settings.qdrant_collection)
        pipeline = DocumentPipeline(store, embedding, index, f"{settings.embedding_base_url}:{settings.embedding_model}")

    if memory is None:
        file_graph = FileGraphStore(store)
        graph = (
            Neo4jGraphStore(
                settings.neo4j_url,
                settings.neo4j_user,
                settings.neo4j_password,
                settings.neo4j_database,
                mirror=file_graph,
            )
            if settings.neo4j_url
            else file_graph
        )
        memory = MemoryService(
            store,
            pipeline.embedding,
            pipeline.index,
            pipeline.embedding_identity,
            graph=graph,
            default_strategy=settings.default_memory_strategy,  # type: ignore[arg-type]
        )

    embedding_settings_service = EmbeddingSettingsService()
    if not injected_pipeline:
        embedding_settings_service.apply_persisted_if_exists(
            store, settings.default_tenant_id, settings.default_tenant_id, pipeline, memory
        )

    experiments = experiments or ExperimentService(store, pipeline, memory)
    knowledge = KnowledgeService(store, pipeline, memory)
    atlas = EmbeddingAtlasService(store, memory)
    mcp_server = MCPServer(memory)
    ingestion = IngestionService(store)

    default_services = TenantServices(
        tenant_id=settings.default_tenant_id,
        store=store,
        ingestion=ingestion,
        pipeline=pipeline,
        memory=memory,
        experiments=experiments,
        knowledge=knowledge,
        atlas=atlas,
        mcp=mcp_server,
    )
    registry = TenantContextRegistry(settings, default_services)
    ent_store = EnterpriseStore(store.root, settings.database_url)
    auth_service = AuthService(settings, ent_store)
    rate_limiter = RateLimiter(redis_url=settings.redis_url, default_rpm=settings.rate_limit_rpm)
    job_queue = JobQueue(ent_store, redis_url=settings.redis_url)
    s3_mirror = S3ObjectMirror(
        endpoint_url=settings.s3_endpoint_url,
        bucket=settings.s3_bucket,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        region=settings.s3_region,
    )
    metrics = MetricsRegistry()

    app = FastAPI(title="Memory Lab Enterprise", version="0.3.0")
    app.state.registry = registry
    app.state.enterprise_store = ent_store
    app.state.job_queue = job_queue
    app.state.metrics = metrics
    app.state.embedding_settings = embedding_settings_service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def enterprise_observability_and_ratelimit(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid4().hex[:16]
        request.state.request_id = request_id
        started = time.perf_counter()

        rl_headers: dict[str, str] = {}
        if request.url.path not in {"/health", "/ready", "/metrics"} and settings.rate_limit_rpm > 0:
            tenant_hint = ""
            raw_key = request.headers.get("X-API-Key", "").strip()
            if raw_key:
                key_row = ent_store.lookup_api_key(raw_key)
                if key_row:
                    tenant_hint = key_row["tenant_id"]
            if not tenant_hint:
                tenant_hint = request.headers.get("X-Tenant-ID", settings.default_tenant_id).strip().lower() or settings.default_tenant_id
            tenant_row = ent_store.get_tenant(tenant_hint)
            limit_rpm = int(tenant_row["rate_limit_rpm"]) if tenant_row else settings.rate_limit_rpm
            allowed, limit, remaining = rate_limiter.check(tenant_hint, limit_rpm=limit_rpm)
            rl_headers = {
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": str(remaining),
            }
            if not allowed:
                metrics.observe_rate_limit(tenant_hint)
                metrics.observe_request(request.method, request.url.path, 429, time.perf_counter() - started)
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded for tenant"},
                    headers={**rl_headers, "Retry-After": "60", "X-Request-ID": request_id},
                )

        response = await call_next(request)
        elapsed = time.perf_counter() - started
        metrics.observe_request(request.method, request.url.path, response.status_code, elapsed)
        response.headers["X-Request-ID"] = request_id
        for hk, hv in rl_headers.items():
            response.headers[hk] = hv

        ctx: AuthContext | None = getattr(request.state, "auth_ctx", None)
        if ctx is not None and request.method in {"POST", "PUT", "DELETE"} and request.url.path not in {"/auth/token"}:
            try:
                ent_store.record_audit(
                    tenant_id=ctx.tenant_id,
                    actor=ctx.actor,
                    role=ctx.role,
                    action=request.method,
                    resource=request.url.path,
                    status_code=response.status_code,
                    request_id=request_id,
                )
            except Exception:
                pass
        return response

    def resolve(request: Request, required_role: Role = "viewer") -> tuple[AuthContext, TenantServices]:
        ctx = auth_service.authenticate(request)
        auth_service.require_role(ctx, required_role)
        request.state.auth_ctx = ctx
        return ctx, registry.get(ctx.tenant_id)

    def resolve_admin(request: Request) -> tuple[AuthContext, TenantServices]:
        """Protect /admin/* even when AUTH_MODE=disabled unless a matching X-Admin-Secret is presented."""
        if settings.auth_mode == "disabled":
            supplied = request.headers.get("X-Admin-Secret", "").strip()
            expected = settings.admin_bootstrap_key.strip()
            if not (expected and supplied and hmac.compare_digest(supplied, expected)):
                raise HTTPException(
                    403,
                    "Admin endpoints are locked when AUTH_MODE=disabled. "
                    "Enable AUTH_MODE=enterprise or provide a valid X-Admin-Secret.",
                )
        return resolve(request, "admin")

    @app.exception_handler(FileNotFoundError)
    async def missing(_request, _exc):
        return JSONResponse(status_code=404, content={"detail": "Artifact not found"})

    @app.exception_handler(ValueError)
    async def invalid(_request, exc):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    # --- Liveness, Readiness & Prometheus Metrics ---

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "memory-lab", "scope": "liveness", "version": "0.3.0"}

    @app.get("/ready")
    def ready():
        tenants = ent_store.list_tenants()
        return {
            "status": "ready",
            "auth_mode": settings.auth_mode,
            "metadata_backend": ent_store.backend_name,
            "s3_mirror_enabled": s3_mirror.enabled,
            "redis_configured": bool(settings.redis_url),
            "neo4j_configured": bool(settings.neo4j_url),
            "tenant_count": len(tenants),
        }

    @app.get("/metrics")
    def prometheus_metrics():
        if not settings.metrics_enabled:
            raise HTTPException(404, "Metrics disabled")
        tenants = ent_store.list_tenants()
        jobs = ent_store.list_jobs(limit=500)
        extra = {
            "memory_lab_tenants_total": float(len(tenants)),
            "memory_lab_jobs_queued": float(sum(1 for j in jobs if j["status"] == "queued")),
            "memory_lab_jobs_running": float(sum(1 for j in jobs if j["status"] == "running")),
            "memory_lab_jobs_completed": float(sum(1 for j in jobs if j["status"] == "completed")),
            "memory_lab_jobs_failed": float(sum(1 for j in jobs if j["status"] == "failed")),
        }
        return PlainTextResponse(
            metrics.render_prometheus(extra_gauges=extra),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    # --- Enterprise Auth, Tenants, API Keys & Audit Logs ---

    @app.post("/auth/token")
    def issue_token(request: Request):
        ctx = auth_service.authenticate(request)
        return auth_service.issue_token_for_context(ctx)

    @app.get("/auth/me")
    def auth_me(request: Request):
        ctx, svc = resolve(request, "viewer")
        tenant = ent_store.get_tenant(ctx.tenant_id) or {"id": ctx.tenant_id, "name": ctx.tenant_id}
        return {
            "tenant": tenant,
            "actor": ctx.actor,
            "role": ctx.role,
            "auth_method": ctx.auth_method,
            "auth_mode": settings.auth_mode,
            "usage": {
                "documents": len(svc.store.documents()),
                "memories": len(svc.memory.list_records(limit=1000)),
            },
        }

    @app.get("/admin/tenants")
    def admin_list_tenants(request: Request):
        resolve_admin(request)
        return ent_store.list_tenants()

    @app.post("/admin/tenants", status_code=201)
    def admin_create_tenant(request: Request, payload: TenantCreateRequest):
        resolve_admin(request)
        return ent_store.ensure_tenant(
            payload.id,
            payload.name,
            plan=payload.plan,
            max_documents=payload.max_documents,
            max_memories=payload.max_memories,
            rate_limit_rpm=payload.rate_limit_rpm,
        )

    @app.get("/admin/api-keys")
    def admin_list_api_keys(request: Request, tenant_id: str | None = Query(None)):
        ctx, _svc = resolve_admin(request)
        return ent_store.list_api_keys(tenant_id=tenant_id or ctx.tenant_id)

    @app.post("/admin/api-keys", status_code=201)
    def admin_create_api_key(request: Request, payload: ApiKeyCreateRequest):
        ctx, _svc = resolve_admin(request)
        target_tenant = payload.tenant_id or ctx.tenant_id
        return ent_store.create_api_key(target_tenant, payload.name, payload.role)

    @app.delete("/admin/api-keys/{key_id}")
    def admin_revoke_api_key(request: Request, key_id: str):
        resolve_admin(request)
        if not ent_store.revoke_api_key(key_id):
            raise HTTPException(404, "API key not found")
        return {"id": key_id, "revoked": True}

    @app.get("/admin/audit")
    def admin_audit_logs(request: Request, limit: int = Query(100, ge=1, le=500)):
        ctx, _svc = resolve_admin(request)
        return ent_store.list_audit_logs(tenant_id=ctx.tenant_id, limit=limit)

    # --- Runtime Embedding Provider Settings ---

    @app.get("/settings/embedding")
    def get_embedding_settings(request: Request):
        _ctx, svc = resolve(request, "viewer")
        return embedding_settings_service.get_public(svc.store, settings, svc.pipeline, svc.memory)

    @app.put("/settings/embedding")
    def update_embedding_settings(request: Request, payload: EmbeddingSettingsUpdate):
        ctx, svc = resolve(request, "editor")
        return embedding_settings_service.update(
            svc.store,
            ctx.tenant_id,
            settings.default_tenant_id,
            payload,
            svc.pipeline,
            svc.memory,
            settings,
        )

    @app.post("/settings/embedding/test")
    def test_embedding_settings(request: Request, payload: EmbeddingTestRequest):
        _ctx, svc = resolve(request, "editor")
        return embedding_settings_service.test_connection(svc.store, payload, svc.pipeline, settings)

    # --- Async Job Queue APIs ---

    @app.post("/jobs/process", status_code=202)
    def enqueue_process_job(request: Request, payload: AsyncProcessJobRequest):
        ctx, svc = resolve(request, "editor")
        # Ensure document exists before enqueueing
        svc.store.document(payload.document_id, payload.version)
        return job_queue.enqueue_document_process(
            tenant_id=ctx.tenant_id,
            document_id=payload.document_id,
            version=payload.version,
            config=payload.config,
        )

    @app.get("/jobs")
    def list_jobs(request: Request, limit: int = Query(50, ge=1, le=200)):
        ctx, _svc = resolve(request, "viewer")
        return ent_store.list_jobs(tenant_id=ctx.tenant_id, limit=limit)

    @app.get("/jobs/{job_id}")
    def get_job(request: Request, job_id: str):
        ctx, _svc = resolve(request, "viewer")
        job = ent_store.get_job(job_id, tenant_id=ctx.tenant_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return job

    # --- Phase 1, 3 & 4: Documents, Pipeline, Enrichment, Stage Invalidation & Filtered/Hybrid Search ---

    @app.get("/documents")
    def documents(request: Request):
        _ctx, svc = resolve(request, "viewer")
        return svc.store.documents()

    @app.post("/documents", status_code=201)
    def upload(request: Request, file: UploadFile = File(...), document_id: UUID | None = None):
        ctx, svc = resolve(request, "editor")
        tenant = ent_store.get_tenant(ctx.tenant_id)
        if tenant and len(svc.store.documents()) >= int(tenant["max_documents"]):
            raise HTTPException(429, "Tenant max_documents quota reached")
        content = file.file.read(settings.max_upload_bytes + 1)
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(413, "File exceeds MAX_UPLOAD_BYTES")
        if not content:
            raise HTTPException(422, "File is empty")
        doc = svc.ingestion.ingest(file.filename or "document.txt", content, str(document_id) if document_id else None)
        if s3_mirror.enabled:
            s3_mirror.put_object(
                f"tenants/{ctx.tenant_id}/documents/{doc.id}/{doc.version}/{doc.filename}",
                content,
                content_type=file.content_type or "application/octet-stream",
            )
        return doc

    @app.delete("/documents/{identity}")
    def delete_document(request: Request, identity: str, version: str | None = Query(None)):
        _ctx, svc = resolve(request, "editor")
        return svc.pipeline.delete_document(identity, version=version)

    @app.get("/documents/{identity}/versions/{version}/source")
    def source(request: Request, identity: str, version: str):
        _ctx, svc = resolve(request, "viewer")
        doc = svc.store.document(identity, version)
        return FileResponse(svc.store.raw_path(doc), filename=doc.filename, media_type="application/octet-stream")

    def _run_background_process(pipeline_svc: DocumentPipeline, identity: str, version: str, config: PipelineConfig):
        try:
            pipeline_svc.process(identity, version, config)
        except Exception:
            # Failure status and error details are already persisted to manifest.json by process()
            pass

    @app.post("/documents/{identity}/versions/{version}/process")
    def process(
        request: Request,
        identity: str,
        version: str,
        config: PipelineConfig,
        background_tasks: BackgroundTasks,
        async_mode: bool = Query(False),
        background: bool = Query(False),
    ):
        ctx, svc = resolve(request, "editor")
        if async_mode:
            svc.store.document(identity, version)
            job = job_queue.enqueue_document_process(
                tenant_id=ctx.tenant_id,
                document_id=identity,
                version=version,
                config=config,
            )
            return JSONResponse(status_code=202, content=job)
        if background:
            initial_manifest = svc.pipeline.prepare_run(identity, version, config)
            background_tasks.add_task(_run_background_process, svc.pipeline, identity, version, config)
            return JSONResponse(status_code=202, content=initial_manifest)
        return svc.pipeline.process(identity, version, config)

    @app.get("/runs")
    def runs(request: Request):
        _ctx, svc = resolve(request, "viewer")
        return [svc.store.get(str(path.relative_to(svc.store.root))) for path in sorted(svc.store.root.glob("runs/*/manifest.json"))]

    @app.delete("/runs/{run_id}")
    def delete_run(request: Request, run_id: str):
        _ctx, svc = resolve(request, "editor")
        return svc.pipeline.delete_run(run_id)

    @app.get("/runs/{run_id}/events")
    async def run_events(request: Request, run_id: str):
        """Server-Sent Events stream reporting real-time pipeline stage progress until completion."""
        _ctx, svc = resolve(request, "viewer")
        manifest_path = svc.store.path(f"runs/{run_id}/manifest.json")
        if not manifest_path.exists():
            raise HTTPException(404, "Pipeline run not found")

        async def event_stream():
            last_sent = ""
            for _ in range(300):
                if await request.is_disconnected():
                    break
                if not manifest_path.exists():
                    break
                try:
                    data = svc.store.get(f"runs/{run_id}/manifest.json")
                except Exception:
                    await asyncio.sleep(0.25)
                    continue
                payload = json.dumps(data, ensure_ascii=False)
                if payload != last_sent:
                    last_sent = payload
                    yield f"data: {payload}\n\n"
                if data.get("status") in {"completed", "failed", "invalidated"}:
                    break
                await asyncio.sleep(0.35)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/runs/{run_id}/{artifact}")
    def artifacts(request: Request, run_id: str, artifact: str):
        _ctx, svc = resolve(request, "viewer")
        if artifact not in {"manifest", "normalized", "chunks", "enriched"}:
            raise HTTPException(404, "Unknown artifact")
        return svc.store.get(f"runs/{run_id}/{artifact}.json")

    @app.post("/runs/{run_id}/invalidate")
    def invalidate_run(request: Request, run_id: str, payload: StageInvalidationRequest):
        _ctx, svc = resolve(request, "editor")
        return svc.pipeline.invalidate(run_id, payload.stages)

    @app.get("/search")
    def search(
        request: Request,
        q: str = Query(min_length=1, max_length=10000),
        limit: int = Query(5, ge=1, le=50),
        mode: SearchMode = Query("dense"),
        document_id: str | None = Query(None),
        version: str | None = Query(None),
        pipeline_run_id: str | None = Query(None),
        dedupe: bool = Query(True),
    ):
        _ctx, svc = resolve(request, "viewer")
        return svc.pipeline.search(
            q,
            limit,
            document_id=document_id,
            version=version,
            pipeline_run_id=pipeline_run_id,
            mode=mode,
            dedupe=dedupe,
        )

    # --- Phase 2 & 3: Agent Memory Foundation, Graph, Temporal Hierarchy & Advanced Strategies ---

    @app.post("/memory/store", status_code=201)
    def memory_store(request: Request, payload: MemoryStoreRequest):
        ctx, svc = resolve(request, "editor")
        tenant = ent_store.get_tenant(ctx.tenant_id)
        if tenant and len(svc.memory.list_records(limit=int(tenant["max_memories"]) + 1)) >= int(tenant["max_memories"]):
            raise HTTPException(429, "Tenant max_memories quota reached")
        return svc.memory.store_memory(
            content=payload.content,
            session_id=payload.session_id,
            role=payload.role,
            strategy=payload.strategy,
            timestamp=payload.timestamp,
            tags=payload.tags,
            source_refs=payload.source_refs,
            metadata=payload.metadata,
        )

    @app.post("/memory/search")
    def memory_search_post(request: Request, payload: MemorySearchRequest):
        _ctx, svc = resolve(request, "viewer")
        return svc.memory.search_memory(
            payload.query,
            strategy=payload.strategy,
            limit=payload.limit,
            session_id=payload.session_id,
            temporal_bucket=payload.temporal_bucket,
        )

    @app.get("/memory/search")
    def memory_search_get(
        request: Request,
        q: str = Query(min_length=1, max_length=10000),
        strategy: MemoryStrategyName | None = Query(None),
        limit: int = Query(5, ge=1, le=50),
        session_id: str | None = Query(None),
        temporal_bucket: str | None = Query(None),
    ):
        _ctx, svc = resolve(request, "viewer")
        return svc.memory.search_memory(
            q,
            strategy=strategy,
            limit=limit,
            session_id=session_id,
            temporal_bucket=temporal_bucket,
        )

    @app.get("/memory/records")
    def memory_records(
        request: Request,
        session_id: str | None = Query(None),
        temporal_bucket: str | None = Query(None),
        limit: int = Query(100, ge=1, le=500),
    ):
        _ctx, svc = resolve(request, "viewer")
        return svc.memory.list_records(session_id=session_id, temporal_bucket=temporal_bucket, limit=limit)

    @app.get("/memory/records/{memory_id}")
    def memory_get(request: Request, memory_id: str):
        _ctx, svc = resolve(request, "viewer")
        return svc.memory.get_memory(memory_id)

    @app.delete("/memory/records/{memory_id}")
    def memory_delete(request: Request, memory_id: str):
        _ctx, svc = resolve(request, "editor")
        deleted = svc.memory.delete_memory(memory_id)
        if not deleted:
            raise HTTPException(404, "Memory record not found")
        return {"memory_id": memory_id, "deleted": True}

    @app.get("/memory/traces")
    def memory_traces(
        request: Request,
        session_id: str | None = Query(None),
        limit: int = Query(100, ge=1, le=500),
    ):
        _ctx, svc = resolve(request, "viewer")
        return svc.memory.list_traces(session_id=session_id, limit=limit)

    @app.get("/memory/temporal")
    def memory_temporal(request: Request):
        _ctx, svc = resolve(request, "viewer")
        return svc.memory.temporal_hierarchy()

    @app.get("/memory/graph")
    def memory_graph(request: Request, entity: str | None = Query(None)):
        _ctx, svc = resolve(request, "viewer")
        return svc.memory.graph_snapshot(entity=entity)

    # --- Phase 2: MCP Adapter (JSON-RPC 2.0 over HTTP, tenant-scoped) ---

    @app.post("/mcp")
    def mcp_endpoint(request: Request, payload: dict[str, Any]):
        method = payload.get("method")
        tool_name = (payload.get("params") or {}).get("name") if method == "tools/call" else None
        required: Role = "editor" if tool_name in {"memory_store", "memory_delete"} else "viewer"
        _ctx, svc = resolve(request, required)
        response = svc.mcp.handle_message(payload)
        if response is None:
            return JSONResponse(status_code=204, content=None)
        return response

    # --- Phase 4: Experimentation, Datasets, Benchmarks & Comparison ---

    @app.get("/experiments/datasets")
    def list_datasets(request: Request):
        _ctx, svc = resolve(request, "viewer")
        return svc.experiments.list_datasets()

    @app.post("/experiments/datasets", status_code=201)
    def create_dataset(request: Request, payload: BenchmarkDatasetCreate):
        _ctx, svc = resolve(request, "editor")
        return svc.experiments.create_dataset(payload)

    @app.get("/experiments/datasets/{dataset_id}")
    def get_dataset(request: Request, dataset_id: str):
        _ctx, svc = resolve(request, "viewer")
        return svc.experiments.get_dataset(dataset_id)

    @app.post("/experiments/run", status_code=201)
    def run_experiment(request: Request, payload: ExperimentRunRequest):
        _ctx, svc = resolve(request, "editor")
        return svc.experiments.run_experiment(payload)

    @app.get("/experiments/reports")
    def list_reports(request: Request):
        _ctx, svc = resolve(request, "viewer")
        return svc.experiments.list_reports()

    @app.get("/experiments/reports/{report_id}")
    def get_report(request: Request, report_id: str):
        _ctx, svc = resolve(request, "viewer")
        return svc.experiments.get_report(report_id)

    @app.get("/experiments/compare")
    def compare_reports(request: Request, ids: str = Query(min_length=1)):
        _ctx, svc = resolve(request, "viewer")
        report_ids = [part.strip() for part in ids.split(",") if part.strip()]
        return svc.experiments.compare_reports(report_ids)

    # --- Unified Knowledge Visualization (Tables, Interactive Graph, Embedding Atlas & Human Search) ---

    @app.get("/knowledge/overview")
    def knowledge_overview(request: Request):
        _ctx, svc = resolve(request, "viewer")
        return svc.knowledge.overview()

    @app.get("/knowledge/table")
    def knowledge_table(
        request: Request,
        view: TableViewName = Query("entities"),
        q: str | None = Query(None),
        entity_type: str | None = Query(None),
        source: KnowledgeSourceFilter = Query("all"),
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ):
        _ctx, svc = resolve(request, "viewer")
        return svc.knowledge.table(
            view=view,
            q=q,
            entity_type=entity_type,
            source=source,
            limit=limit,
            offset=offset,
        )

    @app.get("/knowledge/graph")
    def knowledge_graph(
        request: Request,
        entity: str | None = Query(None),
        source: KnowledgeSourceFilter = Query("all"),
        include_cooccurrence: bool = Query(True),
        max_nodes: int = Query(80, ge=5, le=200),
    ):
        _ctx, svc = resolve(request, "viewer")
        return svc.knowledge.graph(
            entity=entity,
            source=source,
            include_cooccurrence=include_cooccurrence,
            max_nodes=max_nodes,
        )

    @app.get("/knowledge/search")
    def knowledge_search(
        request: Request,
        q: str = Query(min_length=1, max_length=10000),
        scope: KnowledgeSourceFilter = Query("all"),
        doc_mode: SearchMode = Query("hybrid"),
        memory_strategy: MemoryStrategyName = Query("hybrid"),
        temporal_bucket: str | None = Query(None),
        limit: int = Query(10, ge=1, le=50),
    ):
        _ctx, svc = resolve(request, "viewer")
        return svc.knowledge.search(
            q,
            scope=scope,
            doc_mode=doc_mode,
            memory_strategy=memory_strategy,
            temporal_bucket=temporal_bucket,
            limit=limit,
        )

    @app.get("/knowledge/atlas")
    def knowledge_atlas(
        request: Request,
        source: AtlasSourceFilter = Query("all"),
        n_clusters: int | None = Query(None, ge=1, le=12),
        limit: int = Query(500, ge=10, le=2000),
    ):
        _ctx, svc = resolve(request, "viewer")
        return svc.atlas.build_atlas(source=source, n_clusters=n_clusters, limit=limit)

    return app


# Uvicorn uses the factory, so imports do not connect to infrastructure.
