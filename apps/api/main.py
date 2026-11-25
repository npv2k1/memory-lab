from typing import Any
from uuid import UUID
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from qdrant_client import QdrantClient
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
from memory_lab.experiments.service import ExperimentService
from memory_lab.mcp.server import MCPServer
from memory_lab.memory.service import MemoryService
from memory_lab.retrieval.embedding import OpenAIEmbeddings
from memory_lab.storage.filesystem.store import FileStore
from memory_lab.storage.graph.store import FileGraphStore, Neo4jGraphStore
from memory_lab.storage.qdrant.index import QdrantIndex


def create_app(
    settings: Settings | None = None,
    pipeline: DocumentPipeline | None = None,
    memory: MemoryService | None = None,
    experiments: ExperimentService | None = None,
) -> FastAPI:
    settings = settings or Settings()
    store = pipeline.store if pipeline else FileStore(settings.data_dir)
    if pipeline is None:
        embedding = OpenAIEmbeddings(settings.embedding_base_url, settings.embedding_api_key, settings.embedding_model)
        index = QdrantIndex(QdrantClient(url=settings.qdrant_url), settings.qdrant_collection)
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

    experiments = experiments or ExperimentService(store, pipeline, memory)
    mcp_server = MCPServer(memory)
    ingestion = IngestionService(store)

    app = FastAPI(title="Memory Lab", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(FileNotFoundError)
    async def missing(_request, _exc):
        return JSONResponse(status_code=404, content={"detail": "Artifact not found"})

    @app.exception_handler(ValueError)
    async def invalid(_request, exc):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "memory-lab", "scope": "liveness"}

    # --- Phase 1, 3 & 4: Documents, Pipeline, Enrichment, Stage Invalidation & Filtered/Hybrid Search ---

    @app.get("/documents")
    def documents():
        return store.documents()

    @app.post("/documents", status_code=201)
    def upload(file: UploadFile = File(...), document_id: UUID | None = None):
        content = file.file.read(settings.max_upload_bytes + 1)
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(413, "File exceeds MAX_UPLOAD_BYTES")
        if not content:
            raise HTTPException(422, "File is empty")
        return ingestion.ingest(file.filename or "document.txt", content, str(document_id) if document_id else None)

    @app.get("/documents/{identity}/versions/{version}/source")
    def source(identity: str, version: str):
        doc = store.document(identity, version)
        return FileResponse(store.raw_path(doc), filename=doc.filename, media_type="application/octet-stream")

    @app.post("/documents/{identity}/versions/{version}/process")
    def process(identity: str, version: str, config: PipelineConfig):
        return pipeline.process(identity, version, config)

    @app.get("/runs")
    def runs():
        return [store.get(str(path.relative_to(store.root))) for path in sorted(store.root.glob("runs/*/manifest.json"))]

    @app.get("/runs/{run_id}/{artifact}")
    def artifacts(run_id: str, artifact: str):
        if artifact not in {"manifest", "normalized", "chunks", "enriched"}:
            raise HTTPException(404, "Unknown artifact")
        return store.get(f"runs/{run_id}/{artifact}.json")

    @app.post("/runs/{run_id}/invalidate")
    def invalidate_run(run_id: str, request: StageInvalidationRequest):
        return pipeline.invalidate(run_id, request.stages)

    @app.get("/search")
    def search(
        q: str = Query(min_length=1, max_length=10000),
        limit: int = Query(5, ge=1, le=50),
        mode: SearchMode = Query("dense"),
        document_id: str | None = Query(None),
        version: str | None = Query(None),
        pipeline_run_id: str | None = Query(None),
    ):
        return pipeline.search(
            q,
            limit,
            document_id=document_id,
            version=version,
            pipeline_run_id=pipeline_run_id,
            mode=mode,
        )

    # --- Phase 2 & 3: Agent Memory Foundation, Graph, Temporal Hierarchy & Advanced Strategies ---

    @app.post("/memory/store", status_code=201)
    def memory_store(request: MemoryStoreRequest):
        return memory.store_memory(
            content=request.content,
            session_id=request.session_id,
            role=request.role,
            strategy=request.strategy,
            timestamp=request.timestamp,
            tags=request.tags,
            source_refs=request.source_refs,
            metadata=request.metadata,
        )

    @app.post("/memory/search")
    def memory_search_post(request: MemorySearchRequest):
        return memory.search_memory(
            request.query,
            strategy=request.strategy,
            limit=request.limit,
            session_id=request.session_id,
            temporal_bucket=request.temporal_bucket,
        )

    @app.get("/memory/search")
    def memory_search_get(
        q: str = Query(min_length=1, max_length=10000),
        strategy: MemoryStrategyName | None = Query(None),
        limit: int = Query(5, ge=1, le=50),
        session_id: str | None = Query(None),
        temporal_bucket: str | None = Query(None),
    ):
        return memory.search_memory(
            q,
            strategy=strategy,
            limit=limit,
            session_id=session_id,
            temporal_bucket=temporal_bucket,
        )

    @app.get("/memory/records")
    def memory_records(
        session_id: str | None = Query(None),
        temporal_bucket: str | None = Query(None),
        limit: int = Query(100, ge=1, le=500),
    ):
        return memory.list_records(session_id=session_id, temporal_bucket=temporal_bucket, limit=limit)

    @app.get("/memory/records/{memory_id}")
    def memory_get(memory_id: str):
        return memory.get_memory(memory_id)

    @app.delete("/memory/records/{memory_id}")
    def memory_delete(memory_id: str):
        deleted = memory.delete_memory(memory_id)
        if not deleted:
            raise HTTPException(404, "Memory record not found")
        return {"memory_id": memory_id, "deleted": True}

    @app.get("/memory/traces")
    def memory_traces(session_id: str | None = Query(None), limit: int = Query(100, ge=1, le=500)):
        return memory.list_traces(session_id=session_id, limit=limit)

    @app.get("/memory/temporal")
    def memory_temporal():
        return memory.temporal_hierarchy()

    @app.get("/memory/graph")
    def memory_graph(entity: str | None = Query(None)):
        return memory.graph_snapshot(entity=entity)

    # --- Phase 2: MCP Adapter (JSON-RPC 2.0 over HTTP) ---

    @app.post("/mcp")
    def mcp_endpoint(payload: dict[str, Any]):
        response = mcp_server.handle_message(payload)
        if response is None:
            return JSONResponse(status_code=204, content=None)
        return response

    # --- Phase 4: Experimentation, Datasets, Benchmarks & Comparison ---

    @app.get("/experiments/datasets")
    def list_datasets():
        return experiments.list_datasets()

    @app.post("/experiments/datasets", status_code=201)
    def create_dataset(payload: BenchmarkDatasetCreate):
        return experiments.create_dataset(payload)

    @app.get("/experiments/datasets/{dataset_id}")
    def get_dataset(dataset_id: str):
        return experiments.get_dataset(dataset_id)

    @app.post("/experiments/run", status_code=201)
    def run_experiment(request: ExperimentRunRequest):
        return experiments.run_experiment(request)

    @app.get("/experiments/reports")
    def list_reports():
        return experiments.list_reports()

    @app.get("/experiments/reports/{report_id}")
    def get_report(report_id: str):
        return experiments.get_report(report_id)

    @app.get("/experiments/compare")
    def compare_reports(ids: str = Query(min_length=1)):
        report_ids = [part.strip() for part in ids.split(",") if part.strip()]
        return experiments.compare_reports(report_ids)

    return app


# Uvicorn uses the factory, so imports do not connect to infrastructure.
