import math
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from apps.api.main import create_app
from memory_lab.core.config import Settings
from memory_lab.documents.pipeline.service import DocumentPipeline
from memory_lab.experiments.metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k
from memory_lab.storage.filesystem.store import FileStore
from memory_lab.storage.qdrant.index import QdrantIndex


class ConstantEmbeddings:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def test_ir_metrics_exact_values():
    flags = [False, True, False, True]
    assert recall_at_k(flags, total_relevant=2, k=4) == 1.0
    assert recall_at_k(flags, total_relevant=2, k=2) == 0.5
    assert mean_reciprocal_rank(flags, k=4) == 0.5
    expected_ndcg = (1 / math.log2(3) + 1 / math.log2(5)) / (1 / math.log2(2) + 1 / math.log2(3))
    assert abs(ndcg_at_k(flags, total_relevant=2, k=4) - expected_ndcg) < 1e-5


def test_experiments_datasets_runs_and_comparison(tmp_path):
    store = FileStore(tmp_path)
    embeddings = ConstantEmbeddings()
    client = QdrantClient(":memory:")
    index = QdrantIndex(client, "exptest")
    pipeline = DocumentPipeline(store, embeddings, index, "const-embed")

    with TestClient(create_app(Settings(data_dir=tmp_path), pipeline)) as api:
        # Ingest & process two distinct documents
        doc_a = api.post(
            "/documents",
            files={"file": ("qdrant.txt", b"Qdrant provides dense cosine vector search for Memory Lab.")},
        ).json()
        doc_b = api.post(
            "/documents",
            files={"file": ("bm25.txt", b"BM25 sparse retrieval scores lexical term frequencies accurately.")},
        ).json()
        api.post(f"/documents/{doc_a['id']}/versions/{doc_a['version']}/process", json={"parser": "text"})
        api.post(f"/documents/{doc_b['id']}/versions/{doc_b['version']}/process", json={"parser": "text"})

        # Create document benchmark dataset
        ds_resp = api.post(
            "/experiments/datasets",
            json={
                "name": "doc-eval-v1",
                "description": "Compare dense vs sparse vs hybrid",
                "cases": [
                    {"query": "dense cosine vector Qdrant", "relevant_document_ids": [doc_a["id"]]},
                    {"query": "lexical term frequencies BM25", "relevant_document_ids": [doc_b["id"]]},
                ],
            },
        )
        assert ds_resp.status_code == 201
        ds = ds_resp.json()
        assert len(ds["dataset_hash"]) == 64

        rep_dense = api.post(
            "/experiments/run",
            json={"name": "docs-dense", "dataset_id": ds["id"], "target": "documents", "search_mode": "dense", "k": 1},
        ).json()
        rep_sparse = api.post(
            "/experiments/run",
            json={"name": "docs-sparse", "dataset_id": ds["id"], "target": "documents", "search_mode": "sparse", "k": 1},
        ).json()
        rep_hybrid = api.post(
            "/experiments/run",
            json={"name": "docs-hybrid", "dataset_id": ds["id"], "target": "documents", "search_mode": "hybrid", "k": 1},
        ).json()

        assert rep_sparse["metrics"]["recall_at_k"] == 1.0
        assert rep_sparse["metrics"]["mrr"] == 1.0
        assert rep_sparse["metrics"]["ndcg_at_k"] == 1.0
        assert rep_hybrid["metrics"]["recall_at_k"] == 1.0

        # Compare all three reports side by side
        cmp = api.get(
            "/experiments/compare",
            params={"ids": f"{rep_dense['id']},{rep_sparse['id']},{rep_hybrid['id']}"},
        ).json()
        assert cmp["baseline_id"] == rep_dense["id"]
        assert cmp["same_dataset_hash"] is True
        assert len(cmp["reports"]) == 3

        # Also test memory experimentation benchmark across strategies
        m1 = api.post(
            "/memory/store",
            json={"content": "Memory Lab uses Neo4j graph traversal for entities.", "strategy": "hybrid"},
        ).json()
        m2 = api.post(
            "/memory/store",
            json={"content": "ZeroTokenMemory uses BM25 lexical ranking with zero API calls.", "strategy": "zero_token"},
        ).json()

        mem_ds = api.post(
            "/experiments/datasets",
            json={
                "name": "mem-eval-v1",
                "cases": [
                    {"query": "Neo4j graph traversal", "relevant_memory_ids": [m1["id"]]},
                    {"query": "BM25 lexical zero API calls", "relevant_memory_ids": [m2["id"]]},
                ],
            },
        ).json()

        mem_rep = api.post(
            "/experiments/run",
            json={
                "name": "mem-zero-token",
                "dataset_id": mem_ds["id"],
                "target": "memory",
                "memory_strategy": "zero_token",
                "k": 2,
            },
        ).json()
        assert mem_rep["metrics"]["recall_at_k"] == 1.0
        assert mem_rep["metrics"]["mrr"] == 1.0
        assert mem_rep["config_snapshot"]["resolved_memory_strategy"] == "zero_token"
    client.close()
