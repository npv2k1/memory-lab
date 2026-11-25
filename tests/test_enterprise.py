from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from apps.api.main import create_app
from memory_lab.core.config import Settings
from memory_lab.documents.pipeline.service import DocumentPipeline
from memory_lab.storage.filesystem.store import FileStore
from memory_lab.storage.qdrant.index import QdrantIndex


class SimpleEmbeddings:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0 if "alpha" in t.lower() else 0.2, 1.0 if "beta" in t.lower() else 0.2] for t in texts]


def test_enterprise_multitenant_rbac_jwt_ratelimit_jobs_audit_and_metrics(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        auth_mode="enterprise",
        jwt_secret="super-secret-enterprise-jwt-key-32bytes",
        admin_bootstrap_key="bootstrap-root-secret",
        rate_limit_rpm=100,
    )
    store = FileStore(tmp_path)
    embeddings = SimpleEmbeddings()
    client = QdrantClient(":memory:")
    index = QdrantIndex(client, "ent")
    pipeline = DocumentPipeline(store, embeddings, index, "ent-embed")
    app = create_app(settings, pipeline)

    with TestClient(app) as api:
        # 1. Unauthenticated access rejected in enterprise mode
        assert api.get("/documents").status_code == 401
        assert api.get("/health").status_code == 200
        assert api.get("/ready").status_code == 200

        admin_hdr = {"X-Admin-Secret": "bootstrap-root-secret"}

        # 2. Create two isolated tenants
        t_alpha = api.post(
            "/admin/tenants",
            headers=admin_hdr,
            json={"id": "acme-alpha", "name": "Acme Alpha Corp", "max_documents": 100, "rate_limit_rpm": 50},
        )
        assert t_alpha.status_code == 201
        t_beta = api.post(
            "/admin/tenants",
            headers=admin_hdr,
            json={"id": "beta-labs", "name": "Beta Labs", "max_documents": 100, "rate_limit_rpm": 3},
        )
        assert t_beta.status_code == 201

        # 3. Mint RBAC API keys (editor + viewer for Alpha, editor for Beta)
        alpha_editor = api.post(
            "/admin/api-keys",
            headers=admin_hdr,
            json={"tenant_id": "acme-alpha", "name": "alpha-writer", "role": "editor"},
        ).json()
        alpha_viewer = api.post(
            "/admin/api-keys",
            headers=admin_hdr,
            json={"tenant_id": "acme-alpha", "name": "alpha-reader", "role": "viewer"},
        ).json()
        beta_editor = api.post(
            "/admin/api-keys",
            headers=admin_hdr,
            json={"tenant_id": "beta-labs", "name": "beta-writer", "role": "editor"},
        ).json()

        # 4. RBAC enforcement: viewer cannot write or access admin; editor cannot access admin
        assert (
            api.post(
                "/memory/store",
                headers={"X-API-Key": alpha_viewer["api_key"]},
                json={"content": "Viewer must not write"},
            ).status_code
            == 403
        )
        assert api.get("/admin/tenants", headers={"X-API-Key": alpha_editor["api_key"]}).status_code == 403

        # 5. Exchange API key for HS256 JWT Bearer token
        tok = api.post("/auth/token", headers={"X-API-Key": alpha_editor["api_key"]})
        assert tok.status_code == 200
        jwt_hdr = {"Authorization": f"Bearer {tok.json()['access_token']}"}
        me = api.get("/auth/me", headers=jwt_hdr).json()
        assert me["tenant"]["id"] == "acme-alpha"
        assert me["role"] == "editor"

        # 6. Strict multi-tenant isolation (Alpha writes doc + memory; Beta sees ZERO of Alpha's data)
        doc_alpha = api.post(
            "/documents",
            headers=jwt_hdr,
            files={"file": ("alpha_secret.txt", b"Alpha proprietary Qdrant architecture on 2026-11-25.")},
        ).json()

        # Enqueue async document processing job (202 Accepted) and execute via worker queue
        queued = api.post(
            f"/documents/{doc_alpha['id']}/versions/{doc_alpha['version']}/process",
            params={"async_mode": "true"},
            headers=jwt_hdr,
            json={"parser": "text"},
        )
        assert queued.status_code == 202
        job_id = queued.json()["id"]
        assert queued.json()["status"] == "queued"

        executed = app.state.job_queue.pop_and_execute(app.state.registry, worker_id="test-worker-1")
        assert executed is not None
        assert executed["id"] == job_id
        assert executed["status"] == "completed"

        job_status = api.get(f"/jobs/{job_id}", headers=jwt_hdr).json()
        assert job_status["status"] == "completed"
        assert job_status["result"]["status"] == "completed"

        # Alpha stores memory and checks knowledge atlas
        api.post(
            "/memory/store",
            headers=jwt_hdr,
            json={"content": "Alpha secret roadmap uses Neo4j.", "strategy": "hybrid"},
        )
        alpha_ov = api.get("/knowledge/overview", headers={"X-API-Key": alpha_viewer["api_key"]}).json()
        assert alpha_ov["documents_count"] == 1
        assert alpha_ov["memory_records_count"] == 1

        # Beta tenant cannot see Alpha's documents, memories, jobs, or atlas points!
        beta_hdr = {"X-API-Key": beta_editor["api_key"], "X-Tenant-ID": "beta-labs"}
        beta_ov = api.get("/knowledge/overview", headers=beta_hdr).json()
        assert beta_ov["documents_count"] == 0
        assert beta_ov["memory_records_count"] == 0
        assert api.get(f"/jobs/{job_id}", headers=beta_hdr).status_code == 404

        # Cross-tenant header spoofing blocked for non-admin
        assert (
            api.get("/documents", headers={"X-API-Key": beta_editor["api_key"], "X-Tenant-ID": "acme-alpha"}).status_code
            == 403
        )

        # 7. Per-tenant rate limiting triggers 429 on Beta (limit = 3 rpm)
        r_limit = api.get("/documents", headers=beta_hdr)
        assert r_limit.status_code == 429
        assert "X-RateLimit-Limit" in r_limit.headers

        # 8. API key revocation immediately blocks access
        assert api.delete(f"/admin/api-keys/{alpha_viewer['id']}", headers=admin_hdr).status_code == 200
        assert api.get("/documents", headers={"X-API-Key": alpha_viewer["api_key"]}).status_code == 401

        # 9. Audit trail recorded mutating calls with request_id
        audits = api.get("/admin/audit", headers={**admin_hdr, "X-Tenant-ID": "acme-alpha"}).json()
        assert len(audits) >= 3
        assert all(a["tenant_id"] == "acme-alpha" and a["request_id"] for a in audits)

        # 10. Prometheus /metrics exposition works
        prom = api.get("/metrics")
        assert prom.status_code == 200
        assert "memory_lab_http_requests_total" in prom.text
        assert "memory_lab_tenants_total" in prom.text
        assert "memory_lab_jobs_completed" in prom.text
    client.close()
