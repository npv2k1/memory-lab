from uuid import UUID
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from qdrant_client import QdrantClient
from memory_lab.core.config import Settings
from memory_lab.core.models import PipelineConfig
from memory_lab.documents.ingestion.service import IngestionService
from memory_lab.documents.pipeline.service import DocumentPipeline
from memory_lab.retrieval.embedding import OpenAIEmbeddings
from memory_lab.storage.filesystem.store import FileStore
from memory_lab.storage.qdrant.index import QdrantIndex


def create_app(settings: Settings | None = None, pipeline: DocumentPipeline | None = None) -> FastAPI:
    settings = settings or Settings()
    store = pipeline.store if pipeline else FileStore(settings.data_dir)
    if pipeline is None:
        embedding = OpenAIEmbeddings(settings.embedding_base_url, settings.embedding_api_key, settings.embedding_model)
        index = QdrantIndex(QdrantClient(url=settings.qdrant_url), settings.qdrant_collection)
        pipeline = DocumentPipeline(store, embedding, index, f"{settings.embedding_base_url}:{settings.embedding_model}")
    ingestion = IngestionService(store)
    app = FastAPI(title="Memory Lab", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.exception_handler(FileNotFoundError)
    async def missing(_request, _exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"detail": "Artifact not found"})

    @app.exception_handler(ValueError)
    async def invalid(_request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "memory-lab", "scope": "liveness"}

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
        if artifact not in {"manifest", "normalized", "chunks"}:
            raise HTTPException(404, "Unknown artifact")
        return store.get(f"runs/{run_id}/{artifact}.json")

    @app.get("/search")
    def search(q: str = Query(min_length=1, max_length=10000), limit: int = Query(5, ge=1, le=50)):
        return pipeline.search(q, limit)

    return app


# Uvicorn uses the factory, so imports do not connect to infrastructure.
