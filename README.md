# Memory Lab

Nền tảng R&D xử lý tài liệu, bộ nhớ dài hạn cho agent và đánh giá truy xuất (đã triển khai đầy đủ Phase 1–4).

## Khởi chạy

Yêu cầu Docker Compose, hoặc Python 3.12+ (`uv`) và Bun `1.4.4`. Embedding endpoint phải hỗ trợ OpenAI `/v1/embeddings` (các chiến lược `sparse`, `zero_token` và `graph` chạy được không cần embedding server).

```bash
cp .env.example .env
# Điền EMBEDDING_BASE_URL, EMBEDDING_MODEL và EMBEDDING_API_KEY nếu dùng dense/vector.
docker compose up --build
```

Web: http://localhost:4321 · API docs: http://localhost:8000/docs.
Trong Docker, dùng `host.docker.internal` để truy cập embedding server trên máy host. Trong chế độ dev, dùng `localhost`.
Docling tải model lần đầu; cần mạng và có thể mất vài phút. Image backend lớn vì bao gồm Docling/PyTorch.

Chạy development:

```bash
uv sync --locked --extra dev --extra e2e
# Nếu cần parse PDF/DOCX bằng Docling:
# uv sync --locked --extra docling --extra dev
docker compose up -d qdrant
uv run uvicorn apps.api.main:create_app --factory --reload
# Terminal khác:
cd apps/web
bun install --frozen-lockfile
bun run dev
```

Chạy MCP server qua stdio cho agent:

```bash
uv run --no-sync python -m memory_lab.mcp
```

Kiểm tra trong virtualenv:

```bash
uv run --no-sync pytest
```

## Sử dụng

1. **Tab 1 — Documents & Pipeline**:
   - Upload PDF/DOCX/TXT/Markdown/HTML (có thể truyền `document_id` UUID để nhóm version).
   - Chọn parser (`docling`/`text`), chunker (`fixed`/`hybrid`), `chunk_size`, `overlap`, bật/tắt `enrichment`, bấm `Process`.
   - Xem `manifest`, `normalized`, `chunks`, `enriched` và xóa cache từng stage qua `Invalidate`.
   - Tìm kiếm ở chế độ `dense`, `sparse` (BM25) hoặc `hybrid` (RRF), lọc theo `document_id` / `pipeline_run_id`, xem `score_breakdown` và tải nguồn gốc.
2. **Tab 2 — Memory Playground**:
   - Lưu interaction trace theo các chiến lược `hybrid`, `zero_token`, `graph`, `vector`.
   - Tìm kiếm memory theo strategy, `session_id`, `temporal_bucket` (`YYYY`, `YYYY-MM`, `YYYY-MM-DD`).
   - Xem & xóa records, kiểm tra **Temporal Hierarchy** (`year → month → day → session`) và **Knowledge Graph**.
3. **Tab 3 — Experimentation & Comparison**:
   - Tạo benchmark dataset (tự động tính `dataset_hash` cố định).
   - Chạy đánh giá trên `documents` (`dense`/`sparse`/`hybrid`) hoặc `memory` (`zero_token`/`graph`/`vector`/`hybrid`) để đo `Recall@K`, `MRR`, `nDCG@K` và độ trễ.
   - So sánh nhiều báo cáo song song với delta so với baseline.

## API chính

### Documents, Pipeline & Retrieval (Phase 1, 3, 4)

- `POST /documents` multipart `file`; optional `document_id` UUID.
- `GET /documents` danh sách các document version.
- `GET /documents/{id}/versions/{version}/source` tải file gốc.
- `POST /documents/{id}/versions/{version}/process` JSON `{"parser":"docling","chunker":"fixed","chunk_size":1200,"overlap":150,"enrich":true}`.
- `GET /runs`, `GET /runs/{id}/{manifest|normalized|chunks|enriched}`.
- `POST /runs/{id}/invalidate` JSON `{"stages":["parse"|"normalize"|"chunk"|"enrich"|"index"]}`.
- `GET /search?q=...&limit=5&mode=dense|sparse|hybrid&document_id=...&version=...&pipeline_run_id=...`.

### Agent Memory & MCP (Phase 2, 3)

- `POST /memory/store` lưu raw trace và index memory record.
- `POST /memory/search` & `GET /memory/search?q=...&strategy=hybrid|zero_token|graph|vector&limit=5&session_id=...&temporal_bucket=...`.
- `GET /memory/records`, `GET /memory/records/{id}`, `DELETE /memory/records/{id}`.
- `GET /memory/traces` danh sách raw interaction traces bất biến.
- `GET /memory/temporal` phân cấp thời gian `year → month → day → session`.
- `GET /memory/graph?entity=...` snapshot đồ thị thực thể/quan hệ.
- `POST /mcp` JSON-RPC 2.0 MCP endpoint (`initialize`, `tools/list`, `tools/call` với `memory_store`, `memory_search`, `memory_get`, `memory_delete`).

### Experimentation (Phase 4)

- `GET /experiments/datasets`, `POST /experiments/datasets`, `GET /experiments/datasets/{id}`.
- `POST /experiments/run` chạy benchmark và lưu `config_snapshot` + `dataset_hash`.
- `GET /experiments/reports`, `GET /experiments/reports/{id}`.
- `GET /experiments/compare?ids=id1,id2,...` so sánh metrics và delta theo baseline.
- `GET /health` kiểm tra liveness của tiến trình API.

## Boundaries

| Module | Trách nhiệm |
|---|---|
| `apps/api` | HTTP adapter, MCP HTTP endpoint và dependency composition |
| `apps/web` | Documents/Pipeline inspector, Memory Playground, Experimentation & Comparison UI |
| `core` | Models, protocols, environment settings |
| `documents/ingestion` | Nhận bytes, kiểm tra dữ liệu đầu vào và gọi raw storage |
| `documents/parsers` | Docling/text adapter → `ParsedDocument`; giữ page, bbox, heading và source refs |
| `documents/normalization` | Gắn identity/version và kiểm tra block IDs |
| `documents/chunkers` | Fixed character windows hoặc Docling HybridChunker |
| `documents/enrichment` | Non-generative entity, relation, temporal và keyword extraction (không gọi LLM) |
| `documents/indexing` | Cache embeddings và upsert qua `VectorIndex` protocol |
| `documents/pipeline` | Điều phối các stage, manifest, retry, stage invalidation và filtered/hybrid search |
| `memory` | `TraceStore`, `TemporalHierarchy`, `VectorMemory`, `GraphMemory`, `ZeroTokenMemory`, `HybridMemory`, `MemoryService` |
| `mcp` | MCP JSON-RPC 2.0 adapter (stdio + HTTP) cho `memory_store/search/get/delete` |
| `experiments` | Datasets với `dataset_hash`, `Recall@K`/`MRR`/`nDCG@K`, config snapshots và comparison |
| `retrieval` | Embedding provider, Okapi BM25 sparse index và Reciprocal Rank Fusion (RRF) |
| `storage/filesystem` | Source of truth: raw data, metadata, atomic artifacts, traces, records và locks |
| `storage/qdrant` | Rebuildable dense index có metadata payload filtering |
| `storage/graph` | `FileGraphStore` mặc định và `Neo4jGraphStore` HTTP adapter tùy chọn |

## Trạng thái các Phase

1. **Phase 1 — Foundation + Document Core (đã hoàn tất):** contracts → ingestion/raw → parsers → normalization → chunking → artifacts → embeddings/Qdrant → API/web.
2. **Phase 2 — Memory Foundation (đã hoàn tất):** interaction schema/raw traces → `MemoryStrategy` → `VectorMemory`/`GraphMemory` (`FileGraphStore` + optional `Neo4jGraphStore`) → `MemoryService` → MCP adapter (`memory_store/search/get/delete` qua stdio & HTTP).
3. **Phase 3 — Advanced Memory (đã hoàn tất):** non-generative enrichment extractor → Okapi BM25 sparse retrieval + RRF evidence fusion → `TemporalHierarchy` (`year → month → day → session`) → `ZeroTokenMemory` và `HybridMemory` (tự động hạ cấp mượt khi không có embedding server).
4. **Phase 4 — Experimentation (đã hoàn tất):** stage cache & invalidation API → search filters (`document_id`/`version`/`pipeline_run_id`) → dataset/config snapshots → `Recall@K`/`MRR`/`nDCG@K` → Comparison UI.

Chi tiết hợp đồng và ranh giới xử lý dữ liệu: [documents modules](src/memory_lab/documents/README.md).
Hướng dẫn tích hợp vào LLM & AI Agent (MCP, REST API, Tool Calling, Python SDK): [LLM integration guide](docs/llm-integration.md).

## Benchmark trên Google Colab

So sánh trực tiếp 4 chiến lược của Memory Lab (`ZeroToken`, `Graph`, `Vector`, `Hybrid`) với `Naive-Recency`, `RankBM25`, `FAISS-FlatIP`, `ChromaDB` và tùy chọn `Mem0-OSS` bằng notebook [`notebooks/memory_benchmark_colab.ipynb`](notebooks/memory_benchmark_colab.ipynb):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/npv2k1/memory-lab/blob/feat/complete-memory-lab/notebooks/memory_benchmark_colab.ipynb)

## Docker, CI/CD và E2E

[Hướng dẫn delivery](docs/delivery.md): Compose có healthchecks, frontend Nginx proxy `/api`, GitHub Actions integration/browser tests, publish GHCR và manual deploy trên self-hosted runner.
