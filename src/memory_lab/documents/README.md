# Document processing modules

| Module | Input → output | Trách nhiệm |
|---|---|---|
| `ingestion` | filename + bytes → Document | Nhận dữ liệu, kiểm tra rỗng, dùng storage cho hash/version/raw persistence |
| `parsers` | raw file → ParsedDocument | Chuyển TXT/Docling thành blocks trung gian; giữ native artifact sau adapter |
| `normalization` | Document + ParsedDocument → NormalizedDocument | Gắn identity/version, kiểm tra block IDs, bảo toàn nội dung và provenance |
| `chunkers` | NormalizedDocument → chunks | Fixed/Hybrid strategy; giữ source block refs |
| `enrichment` | chunks → enriched evidence (dự kiến) | Điểm mở rộng, chưa được thực thi |
| `indexing` | chunks → embeddings + index | Cache vectors và upsert qua VectorIndex protocol |
| `pipeline` | Document/version + config → manifest | Điều phối stage, run identity, locking và trạng thái lỗi |

`core/contracts.py` chứa parser/chunker/embedding/index protocols. `core/models.py` chứa dữ liệu trung lập với provider. Các service dùng storage tại boundary; pipeline không tự xử lý Docling/Qdrant objects.

Có thể inject `ParsingService` với parser registry, `ChunkingService` với strategy factories, hoặc thay các stage qua keyword arguments của `DocumentPipeline`. Parser mới cần `version` và `parse(path)`; cấu hình/API hiện chỉ công bố các lựa chọn đã hỗ trợ (`text/docling`, `fixed/hybrid`).

Artifacts theo run:

- `parsed.json`: native parser artifact cho Hybrid adapter.
- `parsed_blocks.json`: ParsedDocument để parse stage chạy độc lập với normalization.
- `normalized.json`, `chunks.json`, `vectors.json`, `manifest.json`.

Retry tái sử dụng từng artifact đã hoàn thành, luôn upsert index. Revision pipeline tăng từ 1 lên 2 để không đọc nhầm cache theo contract cũ. Run cũ vẫn đọc được qua inspector; index cũ vẫn được giữ theo chính sách hiện tại.
