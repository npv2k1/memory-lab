# Memory Lab

Nền tảng R&D bộ nhớ dài hạn cho agent. Repository mới, triển khai vertical slice Phase 1; chưa triển khai Phase 2–4.

## Khởi chạy

Yêu cầu Docker Compose, hoặc Python 3.12+ và Bun. Embedding endpoint phải hỗ trợ OpenAI `/v1/embeddings`.

```bash
cp .env.example .env
# Điền EMBEDDING_BASE_URL, EMBEDDING_MODEL và EMBEDDING_API_KEY nếu cần.
docker compose up --build
```

Web: http://localhost:4321 · API docs: http://localhost:8000/docs.
Trong Docker, dùng `host.docker.internal` để truy cập embedding server trên máy host. Trong chế độ dev, dùng `localhost`.
Docling tải model lần đầu; cần mạng và có thể mất vài phút. Image backend lớn vì bao gồm Docling/PyTorch.

Chạy development:

```bash
uv sync --locked --extra docling --extra dev
docker compose up -d qdrant
uv run uvicorn apps.api.main:create_app --factory --reload
# Terminal khác:
cd apps/web
bun install --frozen-lockfile
bun run dev
```

Kiểm tra trong virtualenv:

```bash
uv run pytest
```

## Sử dụng

1. Upload PDF/DOCX/TXT/Markdown/HTML trên web.
2. Chọn parser và chunker, bấm Process. TXT/Markdown hỗ trợ parser `text` để thử không cần Docling.
3. Xem manifest, normalized blocks và chunks trong inspector.
4. Search để xem score, chunk, block references và tải tài liệu gốc.

API tương ứng:

- `POST /documents` multipart `file`; optional `document_id` UUID để tạo version cùng document.
- `GET /documents` danh sách các version.
- `POST /documents/{id}/versions/{version}/process` JSON `{"parser":"docling","chunker":"fixed","chunk_size":1200,"overlap":150}`.
- `GET /runs`, `GET /runs/{id}/{manifest|normalized|chunks}`.
- `GET /search?q=...&limit=5`.
- `GET /documents/{id}/versions/{version}/source`.
- `GET /health` chỉ kiểm tra tiến trình API, không kiểm tra infrastructure.

## Boundaries

| Module | Trách nhiệm |
|---|---|
| `apps/api` | HTTP adapter và dependency composition |
| `apps/web` | Upload, processing, artifact inspector, retrieval playground |
| `core` | Models, protocols, environment settings |
| `documents/parsers` | Docling/text adapter → normalized blocks; giữ page, bbox, heading và source refs |
| `documents/chunkers` | Fixed character windows hoặc Docling HybridChunker |
| `documents/pipeline` | Điều phối, manifest, retry và versioned outputs |
| `storage/filesystem` | Raw data, metadata, atomic artifacts và locks |
| `storage/qdrant` | Rebuildable dense index |
| `retrieval` | Embedding provider adapter |

## Quyết định và giới hạn Phase 1

- Filesystem là dữ liệu gốc. Qdrant không phải bản lưu duy nhất. Lưu normalized/chunks/vectors độc lập để có thể rebuild index.
- Content hash xác định version; upload trùng nội dung trả cùng identity. Truyền document UUID khi muốn chủ động nhóm nhiều version.
- Run ID phụ thuộc document/version, parser version, chunk config, embedding endpoint/model và pipeline revision. Retry cùng config tái sử dụng artifacts và upsert chunk UUID cố định. Thay config tạo run mới.
- Mỗi run giữ snapshot đầy đủ. Chưa chia sẻ stage cache giữa các config khác nhau. Muốn chạy lại stage đã thành công hiện cần tạo config/revision mới; chưa có API invalidation từng stage.
- Search chỉ tìm namespace của embedding endpoint/model hiện tại; giữ evidence từ mọi document version/run đã index. Khi so sánh thí nghiệm cần bổ sung filter run/version ở Phase 4.
- Fixed chunking theo ký tự, có overlap và block references. Hybrid sử dụng Docling artifact qua adapter; không đưa kiểu Docling vào business contracts.
- Processing đồng bộ, chạy trong threadpool của FastAPI; lỗi giữ manifest `failed`. Redis worker/queue và progress realtime chưa triển khai.
- Neo4j/Redis dành cho phase sau, bật bằng `docker compose --profile memory up -d neo4j redis`.
- Đây là môi trường local, không có authentication; ports Compose chỉ bind loopback. Không expose trực tiếp ra Internet.
- Dependencies được khóa bằng `uv.lock` và `apps/web/bun.lock`. Khi benchmark cần lưu thêm model revision/weights và dataset hash.
- Cần thay embedding model name/config khi thay weights trên cùng endpoint; namespace không tự phát hiện weights thay đổi.

## Kế hoạch theo dependency

1. **Foundation + Document Core (hiện tại):** contracts → ingestion/raw → parsers → normalization → chunking → artifacts → embeddings/Qdrant → API/web.
2. **Memory Foundation (chưa làm):** interaction schema/raw traces → MemoryStrategy → VectorMemory/GraphMemory → services → MCP adapter (`memory_store/search/get/delete`).
3. **Advanced Memory (chưa làm):** HybridMemory, ZeroTokenMemory, non-generative extraction, temporal hierarchy, sparse/BM25 và evidence fusion.
4. **Experimentation (chưa làm):** stage cache/invalidation API, run filters, dataset/config snapshots, Recall@K/MRR/nDCG và comparison UI.

Chưa có trước khi setup: toàn bộ source code và infrastructure. Sau setup: Phase 1 đã có code; PDF/Docling và Docker cần được chạy với model/infrastructure thực tế để xác nhận integration. Tests dùng Qdrant local và embedding stub, không đo chất lượng semantic hay thay thế kiểm tra PDF thực tế.

## Kiểm tra tại thời điểm bàn giao

- 2 integration tests passed: FastAPI + Qdrant local + deterministic embedding stub, bao gồm retry/rebuild và failure recovery.
- `bun run build` thành công (Astro 5.18.2).
- Chưa chạy Docker Compose/PDF thực tế: môi trường setup không có Docker và chưa cấu hình embedding model.
