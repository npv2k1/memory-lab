from pathlib import Path
from fastapi.testclient import TestClient
from apps.api.main import create_app
from memory_lab.core.config import Settings
from memory_lab.documents.enrichment.extractor import NonGenerativeExtractor
from memory_lab.documents.enrichment.vi_tokenizer import (
    extract_evidence_sentence,
    is_single_syllable_vi_noise,
    segment_vietnamese,
)
from memory_lab.documents.pipeline.service import DocumentPipeline
from memory_lab.storage.filesystem.store import FileStore
from memory_lab.storage.qdrant.index import QdrantIndex
from qdrant_client import QdrantClient


class FakeEmbedding:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            low = text.lower()
            vectors.append(
                [
                    1.0 if ("nguyên liệu" in low or "nhà máy" in low or "nạp" in low) else 0.05,
                    1.0 if ("docker" in low or "postgresql" in low or "qdrant" in low or "redis" in low) else 0.05,
                    1.0 if ("alice" in low or "minh" in low or "quản lý" in low) else 0.05,
                    0.2,
                ]
            )
        return vectors


def _make_client(tmp_path: Path) -> TestClient:
    store = FileStore(tmp_path)
    pipeline = DocumentPipeline(
        store,
        FakeEmbedding(),
        QdrantIndex(QdrantClient(":memory:"), "ui_redesign_test"),
        "fake:ui-redesign-v1",
    )
    app = create_app(
        settings=Settings(data_dir=tmp_path, auth_mode="disabled", rate_limit_rpm=0),
        pipeline=pipeline,
    )
    return TestClient(app)


def test_vietnamese_segmentation_and_noise_filtering():
    extractor = NonGenerativeExtractor()
    tokens = segment_vietnamese("Quy trình nạp nguyên liệu và kiểm tra chất lượng bằng PostgreSQL")
    assert "quy trình" in tokens
    assert "nạp nguyên liệu" in tokens
    assert "kiểm tra chất lượng" in tokens

    # Single-syllable Vietnamese noise reported in review should be filtered
    for noisy in ["Diễn", "Kiểu", "Ca", "Bảo", "Duy", "Gửi", "Sẵn", "Bị", "Nạp", "Phiếu", "Cân", "Đinh"]:
        assert is_single_syllable_vi_noise(noisy) is True

    # Real names & multi-word Vietnamese entities stay
    assert is_single_syllable_vi_noise("Alice") is False
    assert is_single_syllable_vi_noise("PostgreSQL") is False
    assert is_single_syllable_vi_noise("Memory Lab") is False

    # Sentence-cased Vietnamese compound heads must NOT be extracted as standalone entities
    sample = (
        "Diễn giải quy trình vận hành. Kiểu dữ liệu được lưu trên PostgreSQL. "
        "Bảo trì định kỳ hàng tháng. Duy trì độ ổn định của Docker và Qdrant. "
        "Gửi báo cáo cho Alice. Sẵn sàng nạp nguyên liệu vào dây chuyền."
    )
    entities = extractor.extract_entities(sample)
    names = {e.name for e in entities}
    for bad in {"Diễn", "Kiểu", "Bảo", "Duy", "Gửi", "Sẵn"}:
        assert bad not in names
    assert "PostgreSQL" in names
    assert "Docker" in names
    assert "Qdrant" in names
    assert "Alice" in names

    # Lowercase Vietnamese words starting with đ/ể must NEVER match uppercase proper-noun patterns
    vi_lower_sample = "Đội vận hành xác nhận cảm biến đã đạt chuẩn kiểm tra và dùng Qdrant để lưu trữ."
    vi_lower_names = {e.name for e in extractor.extract_entities(vi_lower_sample)}
    assert "đạt" not in vi_lower_names
    assert "Qdrant để" not in vi_lower_names
    assert "Qdrant" in vi_lower_names

    # Labeled Vietnamese SPO relations
    spo_text = "Alice quản lý Memory Lab. Memory Lab sử dụng PostgreSQL và thuộc về Cloud Ops."
    rels = extractor.extract_relations(spo_text, extractor.extract_entities(spo_text))
    preds = {r.predicate for r in rels}
    assert "manages" in preds
    assert "uses" in preds

    # Evidence sentence quotes the actual sentence containing the entity
    sent = extract_evidence_sentence(sample, "PostgreSQL")
    assert "PostgreSQL" in sent
    assert "Kiểu dữ liệu" in sent


def test_documents_status_detail_topics_ego_graph_and_ask_search(tmp_path: Path):
    client = _make_client(tmp_path)

    doc_content = (
        "# Quy trình nạp nguyên liệu\n\n"
        "Bộ phận vận hành thực hiện quy trình nạp nguyên liệu theo tiêu chuẩn ISO. "
        "Alice quản lý dây chuyền nạp nguyên liệu và báo cáo hàng tuần.\n\n"
        "## Hạ tầng Docker và PostgreSQL\n\n"
        "Memory Lab sử dụng PostgreSQL và Qdrant để lưu trữ dữ liệu tri thức. "
        "Memory Lab chạy trên Docker trong môi trường sản xuất. "
        "Alice sử dụng Memory Lab để tra cứu tài liệu kỹ thuật và theo dõi PostgreSQL.\n\n"
        "## Kiểm tra chất lượng\n\n"
        "Nhóm QA kiểm tra chất lượng nguyên liệu đầu vào trước khi nạp vào bồn chứa. "
        "Mọi sai lệch chất lượng được ghi nhận vào Memory Lab và PostgreSQL."
    ).encode("utf-8")

    up = client.post("/documents", files={"file": ("quy-trinh-he-thong.md", doc_content, "text/markdown")})
    assert up.status_code == 201
    doc = up.json()

    # Before processing, GET /documents includes enriched metadata with latest_run=None
    docs_before = client.get("/documents").json()
    assert len(docs_before) == 1
    assert docs_before[0]["filename"] == "quy-trinh-he-thong.md"
    assert docs_before[0]["extension"] == "md"
    assert docs_before[0]["size_bytes"] > 0
    assert docs_before[0]["latest_run"] is None

    # Process document
    proc = client.post(
        f"/documents/{doc['id']}/versions/{doc['version']}/process",
        json={"parser": "text", "chunker": "fixed", "chunk_size": 300, "overlap": 40},
    )
    assert proc.status_code == 200

    # After processing, GET /documents surfaces latest_run status & chunk_count
    docs_after = client.get("/documents").json()
    assert docs_after[0]["latest_run"] is not None
    assert docs_after[0]["latest_run"]["status"] == "completed"
    assert docs_after[0]["latest_run"]["chunk_count"] >= 1

    # Add a memory so Ask + Topics + Ego Graph combine documents & memory
    mem = client.post(
        "/memory/store",
        json={
            "content": "Alice quản lý Memory Lab và cấu hình PostgreSQL cho dự án quý 4.",
            "session_id": "ops-review",
            "strategy": "hybrid",
            "tags": ["infra", "review"],
        },
    )
    assert mem.status_code == 201

    # 1. Document Detail reading view (TOC, prose chunks, top_entities, topics)
    detail_resp = client.get(f"/documents/{doc['id']}/versions/{doc['version']}/detail")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["document"]["filename"] == "quy-trinh-he-thong.md"
    assert len(detail["chunks"]) >= 1
    assert len(detail["toc"]) >= 1
    assert len(detail["top_entities"]) >= 1
    assert len(detail["topics"]) >= 1

    # 2. Topic Map (c-TF-IDF over Vietnamese phrases)
    topics_resp = client.get("/topics?source=all")
    assert topics_resp.status_code == 200
    topics_data = topics_resp.json()
    assert topics_data["total_items"] >= 2
    assert len(topics_data["topics"]) >= 1
    first_topic = topics_data["topics"][0]
    assert first_topic["label"]
    assert first_topic["item_count"] >= 1
    assert first_topic["share_pct"] > 0
    assert isinstance(first_topic["representative_chunks"], list)
    assert isinstance(first_topic["top_entities"], list)
    if first_topic["top_entities"]:
        assert first_topic["top_entities"][0]["mentions"] >= 1

    # 3. Ego Graph centered on Memory Lab (depth 1, explicit relations only by default)
    ego_resp = client.get("/entities/memory%20lab/neighbors?depth=1&limit=15&include_cooccurrence=false")
    assert ego_resp.status_code == 200
    ego = ego_resp.json()
    assert ego["center"]["id"] == "memory lab"
    assert ego["center"]["hop"] == 0
    assert len(ego["nodes"]) >= 2
    assert all(e["predicate"] != "co_occurs_with" for e in ego["edges"])
    assert any(e["predicate_label"] for e in ego["edges"])

    # Depth 2 expansion & 404 on unknown entity
    ego2 = client.get("/entities/memory%20lab/neighbors?depth=2&limit=15")
    assert ego2.status_code == 200
    assert ego2.json()["depth"] == 2
    assert client.get("/entities/khong-ton-tai-xyz/neighbors").status_code == 404

    # 4. Knowledge Table entities with real quoted evidence_sentence + min_mentions + hide_single_syllable
    tbl = client.get("/knowledge/table?view=entities&min_mentions=2&hide_single_syllable=true").json()
    assert "evidence_sentence" in tbl["columns"]
    assert len(tbl["rows"]) >= 1
    assert all(row["mentions"] >= 2 for row in tbl["rows"])
    assert any(row["evidence_sentence"] for row in tbl["rows"])

    # 5. Unified Ask Search with document_id filter & adjacent chunk context for Reading Drawer
    ask = client.get(f"/knowledge/search?q=PostgreSQL&scope=all&document_id={doc['id']}").json()
    assert len(ask["documents"]) >= 1
    first_hit = ask["documents"][0]
    assert first_hit["filename"] == "quy-trinh-he-thong.md"
    assert "adjacent" in first_hit
    assert "prev_text" in first_hit["adjacent"]
    assert "next_text" in first_hit["adjacent"]
    assert isinstance(first_hit["entities"], list)
