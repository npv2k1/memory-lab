from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from apps.api.main import create_app
from memory_lab.core.config import Settings
from memory_lab.documents.enrichment.extractor import NonGenerativeExtractor
from memory_lab.documents.parsers.adapters import TextParser
from memory_lab.documents.parsers.service import ParsingService
from memory_lab.documents.pipeline.service import DocumentPipeline
from memory_lab.retrieval.fusion import reciprocal_rank_fusion
from memory_lab.storage.filesystem.store import FileStore
from memory_lab.storage.qdrant.index import QdrantIndex


class DeterministicEmbeddings:
    def __init__(self):
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    1.0 if "qdrant" in lower or "vector" in lower else 0.1,
                    1.0 if "neo4j" in lower or "graph" in lower else 0.1,
                    1.0 if "bm25" in lower or "sparse" in lower else 0.1,
                ]
            )
        return vectors


def test_non_generative_extractor_en_and_vi():
    extractor = NonGenerativeExtractor()
    text = (
        "Memory Lab uses Qdrant and Neo4j for hybrid retrieval on 2026-11-25. "
        "Hệ thống lưu trữ dữ liệu từ tháng 11 năm 2026 và hỗ trợ BM25."
    )
    entities, relations, temporal, keywords = extractor.extract_text(text, evidence_id="chunk-1")
    entity_names = {e.normalized_name for e in entities}
    assert {"memory lab", "qdrant", "neo4j", "bm25"}.issubset(entity_names)
    assert any(r.subject == "memory lab" and r.predicate == "uses" and r.object == "qdrant" for r in relations)
    temporal_norms = {t.normalized for t in temporal}
    assert "2026-11-25" in temporal_norms
    assert "2026-11" in temporal_norms
    assert len(keywords) > 0


def test_reciprocal_rank_fusion_orders_consensus_first():
    fused = reciprocal_rank_fusion(
        {
            "dense": [("doc-a", 0.9), ("doc-b", 0.8)],
            "sparse": [("doc-b", 4.5), ("doc-c", 2.1)],
        }
    )
    assert fused[0][0] == "doc-b"
    assert set(fused[0][2].keys()) == {"dense", "sparse", "rrf"}


def test_enrichment_filters_hybrid_search_and_stage_invalidation(tmp_path):
    store = FileStore(tmp_path)
    embeddings = DeterministicEmbeddings()
    client = QdrantClient(":memory:")
    index = QdrantIndex(client, "test")

    class CountingParser(TextParser):
        calls = 0

        def parse(self, path):
            self.calls += 1
            return super().parse(path)

    parser = CountingParser()
    pipeline = DocumentPipeline(store, embeddings, index, "det-embed", parsing=ParsingService(store, {"text": parser}))

    with TestClient(create_app(Settings(data_dir=tmp_path), pipeline)) as api:
        doc1 = api.post(
            "/documents",
            files={"file": ("alpha.txt", b"Memory Lab uses Qdrant for dense vector retrieval on 2026-11-25.")},
        ).json()
        doc2 = api.post(
            "/documents",
            files={"file": ("beta.txt", b"BM25 sparse retrieval handles exact keyword matching in Neo4j.")},
        ).json()

        run1 = api.post(f"/documents/{doc1['id']}/versions/{doc1['version']}/process", json={"parser": "text"}).json()
        run2 = api.post(f"/documents/{doc2['id']}/versions/{doc2['version']}/process", json={"parser": "text"}).json()

        assert run1["status"] == "completed"
        assert run1["enrichment_summary"]["entity_count"] >= 2

        enriched = api.get(f"/runs/{run1['pipeline_run_id']}/enriched")
        assert enriched.status_code == 200
        enriched_json = enriched.json()
        assert any(e["normalized_name"] == "qdrant" for e in enriched_json["entities"])
        assert any(t["normalized"] == "2026-11-25" for t in enriched_json["temporal"])

        # Filtered dense search restricts to doc1
        dense_filtered = api.get("/search", params={"q": "retrieval", "mode": "dense", "document_id": doc1["id"]}).json()
        assert len(dense_filtered) == 1
        assert dense_filtered[0]["chunk"]["document_id"] == doc1["id"]

        # Sparse BM25 search finds exact keyword match without calling embeddings
        calls_before_sparse = embeddings.calls
        sparse_hits = api.get("/search", params={"q": "BM25 keyword matching", "mode": "sparse"}).json()
        assert embeddings.calls == calls_before_sparse
        assert sparse_hits[0]["chunk"]["document_id"] == doc2["id"]
        assert "sparse_bm25" in sparse_hits[0]["score_breakdown"]

        # Hybrid RRF search fuses dense + sparse and respects pipeline_run_id filter
        hybrid_hits = api.get(
            "/search",
            params={"q": "Qdrant vector", "mode": "hybrid", "pipeline_run_id": run1["pipeline_run_id"]},
        ).json()
        assert len(hybrid_hits) == 1
        assert hybrid_hits[0]["chunk"]["pipeline_run_id"] == run1["pipeline_run_id"]
        assert "rrf" in hybrid_hits[0]["score_breakdown"]

        # Stage cache invalidation: invalidate parse forces parser to re-run
        assert parser.calls == 2
        inv = api.post(f"/runs/{run1['pipeline_run_id']}/invalidate", json={"stages": ["parse"]})
        assert inv.status_code == 200
        inv_json = inv.json()
        assert inv_json["status"] == "invalidated"
        assert "parsed_blocks.json" in inv_json["removed_artifacts"]
        assert "enriched.json" in inv_json["removed_artifacts"]

        reprocessed = api.post(
            f"/documents/{doc1['id']}/versions/{doc1['version']}/process", json={"parser": "text"}
        ).json()
        assert reprocessed["status"] == "completed"
        assert parser.calls == 3
    client.close()
