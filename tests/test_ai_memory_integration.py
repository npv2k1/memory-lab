from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from apps.api.main import create_app
from memory_lab.core.config import Settings
from memory_lab.documents.pipeline.service import DocumentPipeline
from memory_lab.memory.aging import DecayParams, retention_score
from memory_lab.memory.scrub import scrub_secrets
from memory_lab.storage.filesystem.store import FileStore
from memory_lab.storage.qdrant.index import QdrantIndex


class MockEmbeddings:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.8, 0.2, 0.5] for _ in texts]


def _make_client(tmp_path):
    store = FileStore(tmp_path)
    qclient = QdrantClient(":memory:")
    index = QdrantIndex(qclient, "aimemtest")
    pipeline = DocumentPipeline(store, MockEmbeddings(), index, "mock-embed")
    app = create_app(Settings(data_dir=tmp_path), pipeline)
    return app, qclient, store


def test_secret_scrubbing_and_aging_feedback_sweep_restore(tmp_path):
    scrubbed, count = scrub_secrets(
        "Use key sk-proj-abcdefghijklmnopqrstuvwxyz1234 and ghp_1234567890abcdefghij123456 with password=supersecretvalue"
    )
    assert count == 3
    assert "sk-proj-" not in scrubbed
    assert "ghp_12345" not in scrubbed
    assert "supersecretvalue" not in scrubbed

    app, qclient, _store = _make_client(tmp_path)
    try:
        with TestClient(app) as api:
            # 1. Secret scrubbed automatically on memory store
            m_secret = api.post(
                "/memory/store",
                json={
                    "content": "Deploy token is ghp_1234567890abcdefghij999999 for staging.",
                    "tier": "episodic",
                    "project": "alpha",
                    "strategy": "zero_token",
                },
            ).json()
            assert "ghp_12345" not in m_secret["content"]
            assert "[REDACTED:github_token]" in m_secret["content"]

            # 2. Store old cold episodic memory, expired TTL memory, pinned memory, and semantic memory
            old_ts = (datetime.now(timezone.utc) - timedelta(days=120)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            past_exp = (datetime.now(timezone.utc) - timedelta(hours=1)).replace(microsecond=0).isoformat().replace("+00:00", "Z")

            m_cold = api.post(
                "/memory/store",
                json={
                    "content": "Temporary debug note from 120 days ago.",
                    "tier": "episodic",
                    "project": "alpha",
                    "timestamp": old_ts,
                    "strategy": "zero_token",
                },
            ).json()
            m_expired = api.post(
                "/memory/store",
                json={
                    "content": "Short-lived sprint reminder with TTL.",
                    "tier": "semantic",
                    "pinned": True,
                    "expires_at": past_exp,
                    "project": "alpha",
                    "strategy": "zero_token",
                },
            ).json()
            m_pinned = api.post(
                "/memory/store",
                json={
                    "content": "Never force push to main branch.",
                    "tier": "episodic",
                    "pinned": True,
                    "timestamp": old_ts,
                    "project": "alpha",
                    "strategy": "zero_token",
                },
            ).json()
            m_semantic = api.post(
                "/memory/store",
                json={
                    "content": "Architecture uses Qdrant and BM25 RRF fusion.",
                    "tier": "semantic",
                    "timestamp": old_ts,
                    "project": "alpha",
                    "strategy": "zero_token",
                },
            ).json()

            # 3. Access reinforcement bumps access_count on GET
            fetched = api.get(f"/memory/records/{m_semantic['id']}").json()
            assert fetched["access_count"] == 1
            assert fetched["last_accessed_at"] is not None

            # 4. Feedback adjusts salience monotonically and drops to floor on stale/wrong
            fb_up = api.post(
                f"/memory/records/{m_semantic['id']}/feedback",
                json={"signal": "helpful", "reason": "Confirmed in prod"},
            ).json()
            assert fb_up["salience_before"] == 1.0
            assert fb_up["salience_after"] == 1.25

            fb_stale = api.post(
                f"/memory/records/{m_cold['id']}/feedback",
                json={"signal": "stale", "reason": "Replaced by new stack"},
            ).json()
            assert fb_stale["salience_after"] == 0.25

            # 5. Forget sweep dry-run identifies cold + TTL expired while exempting pinned & semantic
            dry = api.post("/memory/forget-sweep", json={"dry_run": True, "project": "alpha"}).json()
            cand_ids = {c["id"] for c in dry["candidates"]}
            assert m_cold["id"] in cand_ids
            assert m_expired["id"] in cand_ids
            assert m_pinned["id"] not in cand_ids
            assert m_semantic["id"] not in cand_ids
            assert dry["archived_count"] == 0

            # 6. Execute forget sweep -> archives cold + expired reversibly
            executed = api.post("/memory/forget-sweep", json={"dry_run": False, "project": "alpha"}).json()
            assert executed["archived_count"] == 2
            assert api.get(f"/memory/records/{m_cold['id']}").status_code == 404

            # 7. Restore brings archived memory right back into active retrieval
            restored = api.post(f"/memory/records/{m_cold['id']}/restore")
            assert restored.status_code == 200
            assert restored.json()["archived"] is False
            assert api.get(f"/memory/records/{m_cold['id']}").status_code == 200
    finally:
        qclient.close()


def test_zero_llm_lint_duplicates_and_contradictions(tmp_path):
    app, qclient, _store = _make_client(tmp_path)
    try:
        with TestClient(app) as api:
            # Contradicting functional relations: Bob reports_to Alice vs Bob reports_to Charlie
            api.post("/memory/store", json={"content": "Bob reports to Alice.", "project": "org", "strategy": "graph"})
            api.post("/memory/store", json={"content": "Bob reports to Charlie.", "project": "org", "strategy": "graph"})

            # Near-duplicates
            api.post(
                "/memory/store",
                json={
                    "content": "PostgreSQL connection pool timeout should stay at 30 seconds in production.",
                    "project": "org",
                    "strategy": "zero_token",
                },
            )
            api.post(
                "/memory/store",
                json={
                    "content": "PostgreSQL connection pool timeout must stay at 30 seconds in production.",
                    "project": "org",
                    "strategy": "zero_token",
                },
            )

            lint = api.get("/memory/lint", params={"project": "org"}).json()
            assert len(lint["contradictions"]) >= 1
            assert any(c["subject"] == "bob" and c["predicate"] == "reports_to" for c in lint["contradictions"])
            assert len(lint["near_duplicates"]) >= 1
            assert lint["issue_count"] >= 2
    finally:
        qclient.close()


def test_claim_once_handoffs_and_directed_agent_messaging(tmp_path):
    app, qclient, _store = _make_client(tmp_path)
    try:
        with TestClient(app) as api:
            # --- Handoffs ---
            h1 = api.post(
                "/memory/handoffs",
                json={
                    "project": "backend",
                    "from_agent": "claude-code",
                    "summary": "Finished Qdrant filter deletion; ready for UI integration.",
                    "next_steps": ["Add delete button", "Run E2E tests"],
                    "files_touched": ["apps/api/main.py"],
                },
            )
            assert h1.status_code == 201
            hid = h1.json()["id"]

            listed = api.get("/memory/handoffs", params={"project": "backend", "status": "pending"}).json()
            assert len(listed) == 1 and listed[0]["id"] == hid

            # First accept claims atomically
            accepted = api.post(
                "/memory/handoffs/accept",
                json={"handoff_id": hid, "project": "backend", "agent": "codex"},
            )
            assert accepted.status_code == 200
            assert accepted.json()["status"] == "accepted"
            assert accepted.json()["accepted_by"] == "codex"

            # Second accept of same handoff must fail (claim-once invariant)
            second = api.post(
                "/memory/handoffs/accept",
                json={"handoff_id": hid, "project": "backend", "agent": "cursor"},
            )
            assert second.status_code == 422

            # Cancel works on pending handoff
            h2 = api.post("/memory/handoffs", json={"project": "backend", "summary": "Draft handoff"}).json()
            cancelled = api.post(f"/memory/handoffs/{h2['id']}/cancel").json()
            assert cancelled["status"] == "cancelled"

            # --- Directed Agent Messaging ---
            # Measured claim without evidence MUST fail closed (422)
            bad_measured = api.post(
                "/memory/messages",
                json={
                    "from_project": "frontend",
                    "to_project": "backend",
                    "body": "P95 latency jumped to 800ms.",
                    "nature": "measured",
                },
            )
            assert bad_measured.status_code == 422

            # Measured claim WITH evidence succeeds
            m1 = api.post(
                "/memory/messages",
                json={
                    "from_project": "frontend",
                    "from_agent": "cursor",
                    "to_project": "backend",
                    "to_agent": "codex",
                    "subject": "Add export endpoint",
                    "body": "Please add /v1/export streaming NDJSON.",
                    "nature": "measured",
                    "evidence": "pytest tests/test_review_fixes.py -> 19 passed",
                },
            )
            assert m1.status_code == 201

            # Read-only list never consumes mail
            inbox = api.get("/memory/messages", params={"project": "backend", "box": "inbox"}).json()
            assert len(inbox) == 1

            # Wrong agent cannot pop mail addressed specifically to 'codex'
            wrong_pop = api.post(
                "/memory/messages/pop",
                json={"project": "backend", "agent": "other-agent"},
            ).json()
            assert wrong_pop["claimed"] is False

            # Matching agent pops once; second pop gets nothing
            popped = api.post(
                "/memory/messages/pop",
                json={"project": "backend", "agent": "codex"},
            ).json()
            assert popped["claimed"] is True
            assert popped["message"]["subject"] == "Add export endpoint"

            popped_again = api.post(
                "/memory/messages/pop",
                json={"project": "backend", "agent": "codex"},
            ).json()
            assert popped_again["claimed"] is False

            # Outbox cancel works on pending message
            m2 = api.post(
                "/memory/messages",
                json={"from_project": "frontend", "to_project": "backend", "body": "Never mind"},
            ).json()
            c_res = api.post(
                "/memory/messages/cancel",
                json={"project": "frontend", "message_id": m2["id"]},
            ).json()
            assert c_res["count"] == 1
    finally:
        qclient.close()


def test_durable_markdown_wiki_briefing_and_mcp_tools(tmp_path):
    app, qclient, store = _make_client(tmp_path)
    try:
        with TestClient(app) as api:
            # 1. Path traversal rejected
            bad_path = api.put(
                "/memory/pages",
                json={"path": "../escape.md", "title": "Bad", "content": "Nope"},
            )
            assert bad_path.status_code == 422

            # 2. Write two wikilinked Markdown pages
            p1 = api.put(
                "/memory/pages",
                json={
                    "path": "decisions/embedding-model.md",
                    "title": "Use bge-m3 for Vietnamese + English",
                    "kind": "decision",
                    "project": "core",
                    "pinned": True,
                    "tags": ["embedding", "vietnamese"],
                    "content": "We standardize on Ollama bge-m3. See [[concepts/hybrid-rrf.md]] for fusion details.",
                },
            ).json()
            assert p1["version"] == 1
            assert "concepts/hybrid-rrf.md" in p1["links"]
            # Real Markdown file exists on disk with YAML frontmatter
            md_disk = store.path("wiki/pages/decisions/embedding-model.md").read_text(encoding="utf-8")
            assert md_disk.startswith("---\n")
            assert "Ollama bge-m3" in md_disk

            api.put(
                "/memory/pages",
                json={
                    "path": "concepts/hybrid-rrf.md",
                    "title": "Hybrid RRF Retrieval",
                    "kind": "concept",
                    "project": "core",
                    "content": "Combines Qdrant dense vectors and Okapi BM25 sparse scores.",
                },
            )

            # 3. Patch increments version to v2 and retains v1
            patched = api.patch(
                "/memory/pages",
                json={
                    "path": "decisions/embedding-model.md",
                    "append": "Update: benchmarked 1024-dim vectors with strong VI->EN recall.",
                },
            ).json()
            assert patched["version"] == 2

            # 4. Read returns Markdown, backlinks/related, versions, and bumps access_count
            detail = api.get("/memory/pages/read", params={"path": "decisions/embedding-model.md"}).json()
            assert detail["versions"] == [1, 2]
            assert any(r["path"] == "concepts/hybrid-rrf.md" for r in detail["related"])
            assert detail["page"]["access_count"] == 1

            # 5. Soft-delete removes .md on disk; restore recovers v1 as v3
            deleted = api.delete("/memory/pages", params={"path": "decisions/embedding-model.md"}).json()
            assert deleted["deleted"] is True
            assert not store.path("wiki/pages/decisions/embedding-model.md").exists()

            restored = api.post(
                "/memory/pages/restore",
                json={"path": "decisions/embedding-model.md", "version": 1},
            ).json()
            assert restored["version"] == 3
            assert restored["deleted"] is False
            assert "benchmarked 1024-dim" not in restored["content"]
            assert store.path("wiki/pages/decisions/embedding-model.md").exists()

            # 6. Briefing aggregates durable memories, wiki pages, pending handoffs, unread mail
            api.post("/memory/handoffs", json={"project": "core", "summary": "Verify briefing packet"})
            api.post("/memory/messages", json={"from_project": "ops", "to_project": "core", "body": "Check alerts"})
            brief = api.get("/memory/briefing", params={"project": "core"}).json()
            assert brief["unread_messages_count"] == 1
            assert len(brief["pending_handoffs"]) == 1
            assert len(brief["wiki_pages"]) == 2
            assert "Untrusted recalled context" in brief["rendered_markdown"]

            # 7. MCP JSON-RPC exposes all 23 ai-memory + memory-lab tools and executes them
            tools_res = api.post("/mcp", json={"jsonrpc": "2.0", "id": 10, "method": "tools/list"}).json()
            names = {t["name"] for t in tools_res["result"]["tools"]}
            assert {
                "memory_briefing",
                "memory_feedback",
                "memory_forget_sweep",
                "memory_restore",
                "memory_lint",
                "memory_handoff_begin",
                "memory_handoff_accept",
                "memory_message_send",
                "memory_message_pop",
                "memory_write_page",
                "memory_read_page",
                "memory_restore_page",
            }.issubset(names)

            mcp_brief = api.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 11,
                    "method": "tools/call",
                    "params": {"name": "memory_briefing", "arguments": {"project": "core"}},
                },
            ).json()
            assert mcp_brief["result"]["isError"] is False
            assert mcp_brief["result"]["structuredContent"]["project"] == "core"
    finally:
        qclient.close()
