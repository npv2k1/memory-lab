from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from apps.api.main import create_app
from memory_lab.core.config import Settings
from memory_lab.documents.pipeline.service import DocumentPipeline
from memory_lab.storage.filesystem.store import FileStore
from memory_lab.storage.qdrant.index import QdrantIndex


class SimpleEmbeddings:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [
                1.0 if "qdrant" in t.lower() else 0.1,
                1.0 if "neo4j" in t.lower() else 0.1,
                1.0 if "bm25" in t.lower() else 0.1,
            ]
            for t in texts
        ]


def test_knowledge_overview_table_graph_and_unified_search(tmp_path):
    store = FileStore(tmp_path)
    embeddings = SimpleEmbeddings()
    client = QdrantClient(":memory:")
    index = QdrantIndex(client, "kbtest")
    pipeline = DocumentPipeline(store, embeddings, index, "simple-embed")

    with TestClient(create_app(Settings(data_dir=tmp_path), pipeline)) as api:
        # Ingest & process document
        doc = api.post(
            "/documents",
            files={
                "file": (
                    "kb_arch.txt",
                    b"Memory Lab uses Qdrant and BM25 for hybrid document retrieval since 2026-11-25.",
                )
            },
        ).json()
        api.post(f"/documents/{doc['id']}/versions/{doc['version']}/process", json={"parser": "text"})

        # Store agent memory
        api.post(
            "/memory/store",
            json={
                "content": "Memory Lab connects to Neo4j for entity relation graph reasoning on 2026-11-25.",
                "session_id": "kb-session",
                "strategy": "hybrid",
                "tags": ["graph", "neo4j"],
            },
        )

        # 1. Overview
        ov = api.get("/knowledge/overview").json()
        assert ov["documents_count"] == 1
        assert ov["memory_records_count"] == 1
        assert ov["entities_count"] >= 3
        assert ov["relations_count"] >= 2

        # 2. Tabular API across all 5 views
        for view in ("entities", "relations", "chunks", "memories", "timeline"):
            res = api.get("/knowledge/table", params={"view": view})
            assert res.status_code == 200
            data = res.json()
            assert data["view"] == view
            assert data["total"] >= 1
            assert len(data["rows"]) >= 1

        ent_filtered = api.get("/knowledge/table", params={"view": "entities", "q": "qdrant"}).json()
        assert any(r["id"] == "qdrant" for r in ent_filtered["rows"])

        # 3. Unified Knowledge Graph API (merges document + memory entities & relations)
        graph = api.get("/knowledge/graph").json()
        node_ids = {n["id"] for n in graph["nodes"]}
        assert {"memory lab", "qdrant", "neo4j"}.issubset(node_ids)
        assert graph["stats"]["document_nodes"] >= 1
        assert graph["stats"]["memory_nodes"] >= 1

        sub_graph = api.get("/knowledge/graph", params={"entity": "Neo4j"}).json()
        assert any(n["id"] == "neo4j" for n in sub_graph["nodes"])

        # 4. Unified Human Search API (documents + memories + entities + relations)
        search = api.get("/knowledge/search", params={"q": "Qdrant Neo4j hybrid", "scope": "all"}).json()
        assert search["total_hits"] >= 2
        assert len(search["documents"]) >= 1
        assert len(search["memories"]) >= 1
        assert any(e["id"] == "qdrant" for e in search["matched_entities"])
        assert len(search["matched_relations"]) >= 1
    client.close()
