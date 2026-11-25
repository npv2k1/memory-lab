import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from apps.api.main import create_app
from memory_lab.core.config import Settings
from memory_lab.core.models import Block, NormalizedDocument
from memory_lab.documents.chunkers.strategies import FixedChunker, HybridChunkerAdapter
from memory_lab.documents.enrichment.extractor import NonGenerativeExtractor
from memory_lab.documents.pipeline.service import DocumentPipeline
from memory_lab.experiments.metrics import ndcg_at_k
from memory_lab.storage.filesystem.store import FileStore
from memory_lab.storage.qdrant.index import QdrantIndex


class DeterministicEmbeddings:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    1.0 if "postgresql" in lower or "database" in lower else 0.2,
                    1.0 if "alice" in lower or "phoenix" in lower else 0.2,
                    1.0 if "qdrant" in lower or "vector" in lower else 0.2,
                ]
            )
        return vectors


from types import SimpleNamespace
from memory_lab.documents.chunkers.strategies import FixedChunker, HybridChunkerAdapter, _extract_matching_refs


def test_finding_1_and_8_fixed_chunker_boundaries_and_hybrid_fallback():
    # #8: FixedChunker must not slice mid-word
    text = " ".join(f"architecture_token_{i:02d}" for i in range(40))
    doc = NormalizedDocument(
        document_id="doc-1",
        version="v1",
        blocks=[Block(id="b1", text=text, type="paragraph")],
    )
    chunks = FixedChunker(120, 25).chunk(doc, {}, "run-1")
    assert len(chunks) >= 3
    for chunk in chunks:
        # Every space-separated token in chunk should match a full original token (not sliced mid-word)
        for token in chunk.text.split():
            assert token.startswith("architecture_token_") and len(token) == len("architecture_token_00")

    # #1: _extract_matching_refs resolves self_ref, child cref, and parent cref cleanly
    known = {"#/texts/0", "#/groups/1"}
    item_child = SimpleNamespace(self_ref="#/groups/99", children=[SimpleNamespace(cref="#/texts/0")], parent=None)
    item_parent = SimpleNamespace(self_ref="#/texts/99", children=[], parent=SimpleNamespace(cref="#/groups/1"))
    assert _extract_matching_refs(item_child, known) == ["#/texts/0"]
    assert _extract_matching_refs(item_parent, known) == ["#/groups/1"]
    adapter = HybridChunkerAdapter(chunk_size=800, overlap=100)
    assert adapter.chunk_size == 800 and adapter.overlap == 100


def test_finding_2_benchmark_metrics_bounded_and_deduplicated(tmp_path):
    # Even if flags has more True entries than total_relevant, ndcg_at_k must never exceed 1.0
    assert 0.0 <= ndcg_at_k([True, True, True], total_relevant=1, k=5) <= 1.0

    store = FileStore(tmp_path)
    client = QdrantClient(":memory:")
    index = QdrantIndex(client, "ndcgtest")
    pipeline = DocumentPipeline(store, DeterministicEmbeddings(), index, "det-embed")

    with TestClient(create_app(Settings(data_dir=tmp_path), pipeline)) as api:
        # Upload a multi-chunk document where multiple overlapping chunks contain "PostgreSQL"
        body = "\n\n".join(
            [
                f"Section {i}: PostgreSQL stores relational data reliably with MVCC and WAL logging."
                for i in range(6)
            ]
        ).encode()
        doc = api.post("/documents", files={"file": ("db.txt", body)}).json()
        api.post(
            f"/documents/{doc['id']}/versions/{doc['version']}/process",
            json={"parser": "text", "chunker": "fixed", "chunk_size": 140, "overlap": 30},
        )

        # Case 1: Single substring matched by multiple chunks -> nDCG@5 must be <= 1.0
        # Case 2: Two substrings where only 1 exists -> Recall@5 must be 0.5 (not inflated to 1.0)
        ds = api.post(
            "/experiments/datasets",
            json={
                "name": "ndcg-bounds",
                "cases": [
                    {"query": "PostgreSQL relational MVCC", "relevant_substrings": ["PostgreSQL"]},
                    {"query": "PostgreSQL ClickHouse", "relevant_substrings": ["PostgreSQL", "ClickHouse"]},
                ],
            },
        ).json()
        rep = api.post(
            "/experiments/run",
            json={"name": "sparse-k5", "dataset_id": ds["id"], "target": "documents", "search_mode": "sparse", "k": 5},
        ).json()

        for case_res in rep["case_results"]:
            assert 0.0 <= case_res["recall_at_k"] <= 1.0
            assert 0.0 <= case_res["mrr"] <= 1.0
            assert 0.0 <= case_res["ndcg_at_k"] <= 1.0

        assert rep["case_results"][0]["ndcg_at_k"] == 1.0
        assert rep["case_results"][1]["recall_at_k"] == 0.5
        assert rep["metrics"]["ndcg_at_k"] <= 1.0
    client.close()


def test_finding_3_and_4_entity_relation_extraction_and_multihop_graph_memory(tmp_path):
    extractor = NonGenerativeExtractor()

    # #4: Sentence-initial names, n-gram dedupe, acronym typing, infra recognition, VI stopwords & bigrams
    sample = (
        "Alice manages Acme Corp services deployed on Cloud Run using REST API and PostgreSQL on Ubuntu. "
        "Nguyên rất thích cà phê sữa đá và đang phát triển vdaily."
    )
    entities, relations, _temporal, keywords = extractor.extract_text(sample, evidence_id="ev-1")
    ent_map = {e.normalized_name: e.type for e in entities}

    assert "alice" in ent_map and ent_map["alice"] == "person"
    assert "nguyên" in ent_map
    assert "acme corp" in ent_map and ent_map["acme corp"] == "organization"
    assert "cloud run" in ent_map and ent_map["cloud run"] == "technology"
    assert "ubuntu" in ent_map and "postgresql" in ent_map and "vdaily" in ent_map
    assert ent_map.get("api") == "technology"  # NOT organization
    # Subsumed fragments should NOT pollute entities
    assert "cloud" not in ent_map
    assert "run" not in ent_map
    assert "acme" not in ent_map
    assert "corp" not in ent_map
    # Vietnamese stopwords excluded and compound bigram kept
    assert "tôi" not in keywords and "rất" not in keywords and "và" not in keywords
    assert any("cà phê" in kw for kw in keywords)

    # #3: Multi-hop GraphMemory retrieval (Bob -> reports_to -> Alice -> manages -> Project Phoenix)
    store = FileStore(tmp_path)
    client = QdrantClient(":memory:")
    index = QdrantIndex(client, "graphtest")
    pipeline = DocumentPipeline(store, DeterministicEmbeddings(), index, "det-embed")

    with TestClient(create_app(Settings(data_dir=tmp_path), pipeline)) as api:
        api.post("/memory/store", json={"content": "Bob reports to Alice.", "strategy": "graph"})
        api.post(
            "/memory/store",
            json={"content": "Alice is the project manager of Project Phoenix.", "strategy": "graph"},
        )
        api.post(
            "/memory/store",
            json={"content": "Nguyên deployed vdaily on Ubuntu with PostgreSQL.", "strategy": "graph"},
        )

        # Graph snapshot for Alice must have edges (reports_to & manages)
        alice_graph = api.get("/memory/graph", params={"entity": "Alice"}).json()
        assert alice_graph["edge_count"] >= 2
        predicates = {e["predicate"] for e in alice_graph["edges"]}
        assert "reports_to" in predicates
        assert "manages" in predicates

        # Direct EN + VI natural-language queries
        en_hits = api.post("/memory/search", json={"query": "Who is Bob's manager?", "strategy": "graph", "limit": 3}).json()
        assert len(en_hits) >= 1
        assert any("Alice" in h["record"]["content"] for h in en_hits)

        vi_hits = api.post("/memory/search", json={"query": "quản lý của Bob là ai?", "strategy": "graph", "limit": 3}).json()
        assert len(vi_hits) >= 1
        assert any("Alice" in h["record"]["content"] for h in vi_hits)

        # 2-hop query: Bob's manager -> Alice -> Project Phoenix
        hop2_hits = api.post(
            "/memory/search",
            json={"query": "What project does Bob's manager lead?", "strategy": "graph", "limit": 3},
        ).json()
        assert any("Project Phoenix" in h["record"]["content"] for h in hop2_hits)

        # Unaccented Vietnamese entity query ("Nguyen") matches "Nguyên"
        vi_folded = api.post("/memory/search", json={"query": "Nguyen deployed what?", "strategy": "graph", "limit": 3}).json()
        assert any("vdaily" in h["record"]["content"] for h in vi_folded)
    client.close()


def test_finding_5_7_10_dedupe_background_progress_and_delete(tmp_path):
    store = FileStore(tmp_path)
    client = QdrantClient(":memory:")
    index = QdrantIndex(client, "lifecycletest")
    pipeline = DocumentPipeline(store, DeterministicEmbeddings(), index, "det-embed")

    with TestClient(create_app(Settings(data_dir=tmp_path), pipeline)) as api:
        doc = api.post(
            "/documents",
            files={"file": ("arch.txt", b"PostgreSQL powers transactional metadata in Memory Lab.")},
        ).json()

        # Run 1 (chunk_size=1200) and Run 2 (chunk_size=800) on the SAME document version
        run1 = api.post(
            f"/documents/{doc['id']}/versions/{doc['version']}/process",
            json={"parser": "text", "chunker": "fixed", "chunk_size": 1200, "overlap": 100},
        ).json()
        # #7: Process second run in background mode -> returns 202 and completes with 100% progress
        bg_resp = api.post(
            f"/documents/{doc['id']}/versions/{doc['version']}/process",
            params={"background": "true"},
            json={"parser": "text", "chunker": "fixed", "chunk_size": 800, "overlap": 100},
        )
        assert bg_resp.status_code == 202
        run2_id = bg_resp.json()["pipeline_run_id"]

        # Verify manifest completed with 5 stages & progress == 100
        manifest2 = api.get(f"/runs/{run2_id}/manifest").json()
        assert manifest2["status"] == "completed"
        assert manifest2["progress"] == 100
        assert manifest2["stages"]["parse"] == "completed"
        assert manifest2["stages"]["index"] == "completed"

        # #7: SSE events endpoint returns text/event-stream with final completed state
        sse = api.get(f"/runs/{run2_id}/events")
        assert sse.status_code == 200
        assert "text/event-stream" in sse.headers["content-type"]
        assert '"status": "completed"' in sse.text

        # #5: Default search deduplicates across both runs -> only 1 hit (not 2 duplicates!)
        deduped_dense = api.get("/search", params={"q": "PostgreSQL", "mode": "dense", "limit": 5}).json()
        deduped_sparse = api.get("/search", params={"q": "PostgreSQL", "mode": "sparse", "limit": 5}).json()
        assert len(deduped_dense) == 1
        assert len(deduped_sparse) == 1

        # With dedupe=false, both runs are returned
        raw_sparse = api.get("/search", params={"q": "PostgreSQL", "mode": "sparse", "limit": 5, "dedupe": "false"}).json()
        assert len(raw_sparse) == 2

        # #10: Delete run1 cleans its artifacts & index entries
        del_run = api.delete(f"/runs/{run1['pipeline_run_id']}")
        assert del_run.status_code == 200
        assert del_run.json()["deleted"] is True
        assert api.get(f"/runs/{run1['pipeline_run_id']}/manifest").status_code == 404

        # #10: Delete document removes document, remaining run2, Qdrant points, and BM25 entries
        del_doc = api.delete(f"/documents/{doc['id']}")
        assert del_doc.status_code == 200
        assert del_doc.json()["deleted"] is True
        assert run2_id in del_doc.json()["deleted_runs"]
        assert api.get("/documents").json() == []
        assert api.get("/runs").json() == []
        assert api.get("/search", params={"q": "PostgreSQL", "mode": "dense"}).json() == []
        assert api.get("/search", params={"q": "PostgreSQL", "mode": "sparse"}).json() == []
    client.close()


def test_finding_6_and_12_admin_lockdown_and_mcp_protocol_negotiation(tmp_path):
    store = FileStore(tmp_path)
    client = QdrantClient(":memory:")
    index = QdrantIndex(client, "sectest")
    pipeline = DocumentPipeline(store, DeterministicEmbeddings(), index, "det-embed")

    # #6: When AUTH_MODE=disabled, /admin/* returns 403 unless X-Admin-Secret matches ADMIN_BOOTSTRAP_KEY
    settings = Settings(data_dir=tmp_path, auth_mode="disabled", admin_bootstrap_key="secret-admin-key")
    with TestClient(create_app(settings, pipeline)) as api:
        assert api.get("/admin/tenants").status_code == 403
        assert api.post("/admin/api-keys", json={"name": "hack", "role": "admin"}).status_code == 403
        assert api.get("/admin/audit").status_code == 403
        assert api.get("/admin/tenants", headers={"X-Admin-Secret": "wrong"}).status_code == 403
        assert api.get("/admin/tenants", headers={"X-Admin-Secret": "secret-admin-key"}).status_code == 200

        # #12: MCP initialize negotiates protocol version and syncs version to 0.3.0
        init_old = api.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
        ).json()
        assert init_old["result"]["protocolVersion"] == "2024-11-05"
        assert init_old["result"]["serverInfo"]["version"] == "0.3.0"

        init_new = api.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
        ).json()
        assert init_new["result"]["protocolVersion"] == "2025-06-18"

        init_unknown = api.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 3, "method": "initialize", "params": {"protocolVersion": "2099-01-01"}},
        ).json()
        assert init_unknown["result"]["protocolVersion"] == "2025-03-26"
    client.close()


def test_runtime_embedding_provider_settings_hotswap_and_persistence(tmp_path):
    # Spin up a tiny local mock OpenAI-compatible /v1/embeddings server
    class MockEmbedHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            inputs = body.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]
            dim = 4 if body.get("model") == "bge-m3" else 3
            payload = {
                "object": "list",
                "data": [{"object": "embedding", "index": i, "embedding": [0.5] * dim} for i in range(len(inputs))],
            }
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), MockEmbedHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        mock_url = f"http://127.0.0.1:{port}/v1"
        settings = Settings(
            data_dir=tmp_path,
            qdrant_url=":memory:",
            embedding_base_url="http://localhost:1234/v1",
            embedding_model="nomic-embed-text",
        )
        with TestClient(create_app(settings)) as api:
            # 1. GET returns current config + presets, without raw keys
            initial = api.get("/settings/embedding").json()
            assert initial["model"] == "nomic-embed-text"
            assert initial["source"] == "env"
            assert any(p["id"] == "ollama-bge-m3" for p in initial["presets"])
            prev_ns = initial["document_namespace"]

            # 2. POST /settings/embedding/test verifies live connection + dimension
            test_ok = api.post(
                "/settings/embedding/test",
                json={"base_url": mock_url, "model": "bge-m3", "api_key": "sk-test-1234567890"},
            ).json()
            assert test_ok["ok"] is True
            assert test_ok["dimension"] == 4
            assert test_ok["latency_ms"] >= 0

            # 3. PUT /settings/embedding persists, masks API key, and hot-swaps namespaces
            updated = api.put(
                "/settings/embedding",
                json={
                    "provider": "ollama",
                    "base_url": mock_url,
                    "model": "bge-m3",
                    "api_key": "sk-secret-value-9876",
                },
            ).json()
            assert updated["model"] == "bge-m3"
            assert updated["source"] == "persisted"
            assert updated["has_api_key"] is True
            assert updated["api_key_masked"] == "sk-***9876"
            assert "sk-secret-value-9876" not in json.dumps(updated)
            assert updated["namespace_changed"] is True
            assert updated["document_namespace"] != prev_ns

            # 4. Subsequent PUT without api_key preserves existing key
            updated2 = api.put(
                "/settings/embedding",
                json={"provider": "ollama", "base_url": mock_url, "model": "bge-m3"},
            ).json()
            assert updated2["has_api_key"] is True
            assert updated2["api_key_masked"] == "sk-***9876"

            # 5. Hot-swapped provider works immediately for document processing & memory storage!
            doc = api.post("/documents", files={"file": ("vi.txt", "Hệ thống Memory Lab hỗ trợ bge-m3.".encode())}).json()
            proc = api.post(f"/documents/{doc['id']}/versions/{doc['version']}/process", json={"parser": "text"}).json()
            assert proc["status"] == "completed"
            assert proc["config"]["embedding_namespace"] == updated["document_namespace"]

            # 6. clear_api_key=True clears stored key
            cleared = api.put(
                "/settings/embedding",
                json={"provider": "ollama", "base_url": mock_url, "model": "bge-m3", "clear_api_key": True},
            ).json()
            assert cleared["has_api_key"] is False
            assert cleared["api_key_masked"] == ""

        # 7. Restarting create_app restores persisted settings/embedding.json automatically
        with TestClient(create_app(settings)) as api2:
            restored = api2.get("/settings/embedding").json()
            assert restored["model"] == "bge-m3"
            assert restored["base_url"] == mock_url
            assert restored["source"] == "persisted"
    finally:
        server.shutdown()
        server.server_close()
