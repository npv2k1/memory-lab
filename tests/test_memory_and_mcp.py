from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from apps.api.main import create_app
from memory_lab.core.config import Settings
from memory_lab.documents.pipeline.service import DocumentPipeline
from memory_lab.storage.filesystem.store import FileStore
from memory_lab.storage.qdrant.index import QdrantIndex


class TopicEmbeddings:
    def __init__(self, *, fail: bool = False):
        self.calls = 0
        self.fail = fail

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("embedding server offline")
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    1.0 if any(k in lower for k in ("qdrant", "vector", "dense")) else 0.05,
                    1.0 if any(k in lower for k in ("neo4j", "graph", "entity")) else 0.05,
                    1.0 if any(k in lower for k in ("redis", "cache", "worker")) else 0.05,
                ]
            )
        return vectors


def test_memory_strategies_temporal_graph_and_delete(tmp_path):
    store = FileStore(tmp_path)
    embeddings = TopicEmbeddings()
    client = QdrantClient(":memory:")
    index = QdrantIndex(client, "memtest")
    pipeline = DocumentPipeline(store, embeddings, index, "topic-embed")

    with TestClient(create_app(Settings(data_dir=tmp_path), pipeline)) as api:
        # 1. ZeroTokenMemory stores & searches without calling embeddings
        r_zero = api.post(
            "/memory/store",
            json={
                "content": "ZeroTokenMemory relies on BM25 and TemporalHierarchy in 2026-10-15.",
                "session_id": "s-zero",
                "strategy": "zero_token",
                "timestamp": "2026-10-15T09:00:00Z",
                "tags": ["baseline", "bm25"],
            },
        )
        assert r_zero.status_code == 201
        zero_rec = r_zero.json()
        assert embeddings.calls == 0

        zero_hits = api.post(
            "/memory/search",
            json={"query": "BM25 TemporalHierarchy 2026-10", "strategy": "zero_token", "limit": 5},
        ).json()
        assert embeddings.calls == 0
        assert zero_hits[0]["record"]["id"] == zero_rec["id"]
        assert "sparse_bm25" in zero_hits[0]["score_breakdown"]

        # 2. GraphMemory captures entities & relations and retrieves via 1-hop / 2-hop graph
        r_graph1 = api.post(
            "/memory/store",
            json={
                "content": "Memory Lab uses Neo4j for entity relation traversal.",
                "session_id": "s-graph",
                "strategy": "graph",
                "timestamp": "2026-11-20T10:00:00Z",
            },
        ).json()
        r_graph2 = api.post(
            "/memory/store",
            json={
                "content": "Neo4j connects to Redis cache for session state.",
                "session_id": "s-graph",
                "strategy": "graph",
                "timestamp": "2026-11-21T10:00:00Z",
            },
        ).json()
        graph_hits = api.get("/memory/search", params={"q": "Neo4j", "strategy": "graph", "limit": 5}).json()
        hit_ids = {h["record"]["id"] for h in graph_hits}
        assert r_graph1["id"] in hit_ids
        assert r_graph2["id"] in hit_ids

        # Graph snapshot reflects extracted entities and edges
        snap = api.get("/memory/graph", params={"entity": "Neo4j"}).json()
        assert snap["node_count"] >= 2
        assert snap["edge_count"] >= 1

        # 3. VectorMemory & HybridMemory with Qdrant dense embeddings
        r_vec = api.post(
            "/memory/store",
            json={
                "content": "VectorMemory indexes Qdrant dense embeddings on 2026-11-25.",
                "session_id": "s-vec",
                "strategy": "vector",
                "timestamp": "2026-11-25T12:00:00Z",
            },
        ).json()
        assert embeddings.calls == 1

        vec_hits = api.get("/memory/search", params={"q": "Qdrant dense", "strategy": "vector"}).json()
        assert vec_hits[0]["record"]["id"] == r_vec["id"]

        hybrid_hits = api.get(
            "/memory/search",
            params={"q": "Qdrant dense 2026-11-25", "strategy": "hybrid", "temporal_bucket": "2026-11"},
        ).json()
        assert hybrid_hits[0]["record"]["id"] == r_vec["id"]
        assert "rrf" in hybrid_hits[0]["score_breakdown"]

        # 4. Temporal hierarchy has year -> month -> day -> session rollups
        temporal = api.get("/memory/temporal").json()
        assert any(node["bucket"] == "2026" for node in temporal["year"])
        assert any(node["bucket"] == "2026-11" for node in temporal["month"])
        assert any(node["bucket"] == "2026-11-25" for node in temporal["day"])

        # 5. Raw traces remain recorded
        traces = api.get("/memory/traces").json()
        assert len(traces) == 4

        # 6. Delete removes memory from active get/search/graph
        del_resp = api.delete(f"/memory/records/{r_vec['id']}")
        assert del_resp.status_code == 200
        assert api.get(f"/memory/records/{r_vec['id']}").status_code == 404
    client.close()


def test_hybrid_memory_degrades_gracefully_without_embeddings(tmp_path):
    store = FileStore(tmp_path)
    embeddings = TopicEmbeddings(fail=True)
    client = QdrantClient(":memory:")
    index = QdrantIndex(client, "memoffline")
    pipeline = DocumentPipeline(store, embeddings, index, "offline-embed")

    with TestClient(create_app(Settings(data_dir=tmp_path), pipeline)) as api:
        stored = api.post(
            "/memory/store",
            json={"content": "HybridMemory uses FastAPI and Qdrant safely offline.", "strategy": "hybrid"},
        )
        assert stored.status_code == 201
        hits = api.get("/memory/search", params={"q": "FastAPI Qdrant", "strategy": "hybrid"}).json()
        assert len(hits) == 1
        assert hits[0]["record"]["id"] == stored.json()["id"]
    client.close()


def test_mcp_jsonrpc_lifecycle(tmp_path):
    store = FileStore(tmp_path)
    embeddings = TopicEmbeddings()
    client = QdrantClient(":memory:")
    index = QdrantIndex(client, "mcptest")
    pipeline = DocumentPipeline(store, embeddings, index, "mcp-embed")

    with TestClient(create_app(Settings(data_dir=tmp_path), pipeline)) as api:
        init = api.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}).json()
        assert init["result"]["serverInfo"]["name"] == "memory-lab"

        tools = api.post("/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}).json()
        tool_names = {t["name"] for t in tools["result"]["tools"]}
        assert tool_names == {"memory_store", "memory_search", "memory_get", "memory_delete"}

        stored = api.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "memory_store",
                    "arguments": {
                        "content": "MCP integrates Memory Lab with external coding agents.",
                        "session_id": "mcp-session",
                        "strategy": "hybrid",
                    },
                },
            },
        ).json()
        assert stored["result"]["isError"] is False
        memory_id = stored["result"]["structuredContent"]["id"]

        searched = api.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "memory_search", "arguments": {"query": "coding agents MCP", "limit": 3}},
            },
        ).json()
        assert searched["result"]["isError"] is False
        assert searched["result"]["structuredContent"][0]["record"]["id"] == memory_id

        fetched = api.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "memory_get", "arguments": {"memory_id": memory_id}},
            },
        ).json()
        assert fetched["result"]["structuredContent"]["id"] == memory_id

        deleted = api.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {"name": "memory_delete", "arguments": {"memory_id": memory_id}},
            },
        ).json()
        assert deleted["result"]["structuredContent"]["deleted"] is True

        missing = api.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {"name": "memory_get", "arguments": {"memory_id": memory_id}},
            },
        ).json()
        assert missing["result"]["isError"] is True
    client.close()


def test_mcp_stdio_transport(tmp_path, monkeypatch):
    import io
    import json
    from memory_lab.mcp.__main__ import run_stdio
    from memory_lab.mcp.server import MCPServer
    from memory_lab.memory.service import MemoryService

    store = FileStore(tmp_path)
    embeddings = TopicEmbeddings()
    client = QdrantClient(":memory:")
    index = QdrantIndex(client, "mcpstdio")
    memory = MemoryService(store, embeddings, index, "stdio-embed")
    server = MCPServer(memory)

    requests = "\n".join(
        [
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "memory_store",
                        "arguments": {"content": "Stdio MCP memory test with BM25.", "strategy": "zero_token"},
                    },
                }
            ),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "memory_search",
                        "arguments": {"query": "Stdio BM25", "strategy": "zero_token"},
                    },
                }
            ),
        ]
    ) + "\n"

    stdout = io.StringIO()
    monkeypatch.setattr("sys.stdin", io.StringIO(requests))
    monkeypatch.setattr("sys.stdout", stdout)
    run_stdio(server)

    lines = [json.loads(line) for line in stdout.getvalue().strip().splitlines()]
    assert len(lines) == 3  # notification produces no response
    assert lines[0]["id"] == 1
    assert lines[1]["result"]["isError"] is False
    stored_id = lines[1]["result"]["structuredContent"]["id"]
    assert lines[2]["result"]["structuredContent"][0]["record"]["id"] == stored_id
    client.close()

