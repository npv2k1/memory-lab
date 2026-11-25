# Document processing modules

| Module | Input → output | Trách nhiệm |
|---|---|---|
| `ingestion` | filename + bytes → Document | Nhận dữ liệu, kiểm tra rỗng, dùng storage cho hash/version/raw persistence |
| `parsers` | raw file → ParsedDocument | Chuyển TXT/Docling thành blocks trung gian; giữ native artifact sau adapter |
| `normalization` | Document + ParsedDocument → NormalizedDocument | Gắn identity/version, kiểm tra block IDs, bảo toàn nội dung và provenance |
| `chunkers` | NormalizedDocument → chunks | Fixed/Hybrid strategy; giữ source block refs |
| `enrichment` | chunks → EnrichmentArtifact | Trích xuất non-generative entities, relations, temporal anchors và keywords |
| `indexing` | chunks → embeddings + index | Cache vectors và upsert qua `VectorIndex` protocol (kèm BM25 sparse index) |
| `pipeline` | Document/version + config → manifest | Điều phối stage, run identity, locking, stage invalidation và filtered/hybrid search |

`core/contracts.py` chứa parser/chunker/enrichment/embedding/vector/sparse/graph/memory protocols. `core/models.py` chứa dữ liệu trung lập với provider. Các service dùng storage tại boundary; pipeline không tự xử lý Docling/Qdrant objects.

Artifacts theo run:

- `parsed.json`: native parser artifact cho Hybrid adapter.
- `parsed_blocks.json`: `ParsedDocument` để parse stage chạy độc lập với normalization.
- `normalized.json`, `chunks.json`, `enriched.json`, `vectors.json`, `manifest.json`.

Retry tái sử dụng từng artifact đã hoàn thành, luôn upsert index. Có thể xóa cache một stage và toàn bộ downstream stages bằng `POST /runs/{run_id}/invalidate` (`parse`, `normalize`, `chunk`, `enrich`, `index`). Revision pipeline hiện tại là `3`.
