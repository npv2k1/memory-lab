# Enrichment

Stage sau chunking và trước indexing, trích xuất **non-generative** (không gọi LLM):

- `NonGenerativeExtractor` ([extractor.py](extractor.py)): trích xuất entities (công nghệ, tổ chức, danh từ riêng), relations (mẫu vị ngữ Anh/Việt + đồng xuất hiện theo câu), temporal anchors (`YYYY-MM-DD`, `DD/MM/YYYY`, `YYYY-MM`, tháng/năm, mốc tương đối) và keywords.
- `EnrichmentService` ([service.py](service.py)): cache kết quả tại `runs/{run_id}/enriched.json` và giữ nguyên `chunk_id` + `block_ids` provenance.

Module này cũng được tái sử dụng bởi `GraphMemory`, `ZeroTokenMemory` và `HybridMemory` trong `memory_lab.memory`.
