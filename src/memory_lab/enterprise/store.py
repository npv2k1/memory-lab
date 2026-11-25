from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import secrets
import sqlite3
import threading
from typing import Any
from uuid import uuid4
from memory_lab.core.models import utc_now_iso


def hash_secret(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


SCHEMA_SQL = [
    """
    CREATE TABLE IF NOT EXISTS tenants (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        plan TEXT NOT NULL DEFAULT 'enterprise',
        max_documents INTEGER NOT NULL DEFAULT 10000,
        max_memories INTEGER NOT NULL DEFAULT 100000,
        rate_limit_rpm INTEGER NOT NULL DEFAULT 600,
        status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS api_keys (
        id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        name TEXT NOT NULL,
        role TEXT NOT NULL,
        key_prefix TEXT NOT NULL,
        key_hash TEXT NOT NULL UNIQUE,
        revoked INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        job_type TEXT NOT NULL,
        status TEXT NOT NULL,
        progress INTEGER NOT NULL DEFAULT 0,
        payload_json TEXT NOT NULL,
        result_json TEXT,
        error TEXT,
        worker_id TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS audit_logs (
        id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        actor TEXT NOT NULL,
        role TEXT NOT NULL,
        action TEXT NOT NULL,
        resource TEXT NOT NULL,
        status_code INTEGER NOT NULL,
        request_id TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
]


class EnterpriseStore:
    """ACID relational metadata store for tenants, API keys, async jobs, and audit logs.

    Uses PostgreSQL via psycopg when DATABASE_URL is configured and psycopg is installed;
    otherwise uses SQLite in WAL mode inside <data_dir>/enterprise.db.
    """

    def __init__(self, data_dir: Path, database_url: str = ""):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.database_url = database_url.strip()
        self._lock = threading.RLock()
        self._pg = False
        self._psycopg: Any = None

        if self.database_url.startswith(("postgres://", "postgresql://")):
            try:
                import psycopg  # type: ignore[import-not-found]
                self._psycopg = psycopg
                self._pg = True
            except ImportError:
                self._pg = False

        self.sqlite_path = self.data_dir / "enterprise.db"
        self._init_schema()
        self.ensure_tenant("default", "Default Workspace", plan="enterprise")

    @property
    def backend_name(self) -> str:
        return "postgresql" if self._pg else "sqlite"

    def _ph(self, query: str) -> str:
        return query.replace("?", "%s") if self._pg else query

    @contextmanager
    def _conn(self):
        with self._lock:
            if self._pg:
                conn = self._psycopg.connect(self.database_url)
                try:
                    yield conn
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()
            else:
                conn = sqlite3.connect(str(self.sqlite_path), timeout=30)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                try:
                    yield conn
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()

    def _fetchall(self, conn: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        cur = conn.cursor()
        cur.execute(self._ph(sql), params)
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        cur.close()
        return [dict(zip(cols, row, strict=True)) for row in rows]

    def _fetchone(self, conn: Any, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        rows = self._fetchall(conn, sql, params)
        return rows[0] if rows else None

    def _execute(self, conn: Any, sql: str, params: tuple = ()) -> int:
        cur = conn.cursor()
        cur.execute(self._ph(sql), params)
        count = cur.rowcount
        cur.close()
        return count

    def _init_schema(self) -> None:
        with self._conn() as conn:
            for ddl in SCHEMA_SQL:
                self._execute(conn, ddl)

    # --- Tenants ---

    def ensure_tenant(
        self,
        tenant_id: str,
        name: str,
        *,
        plan: str = "enterprise",
        max_documents: int = 10000,
        max_memories: int = 100000,
        rate_limit_rpm: int = 600,
    ) -> dict[str, Any]:
        safe_id = tenant_id.strip().lower()
        if not safe_id or "/" in safe_id or ".." in safe_id:
            raise ValueError("Invalid tenant_id")
        with self._conn() as conn:
            existing = self._fetchone(conn, "SELECT * FROM tenants WHERE id = ?", (safe_id,))
            if existing:
                return existing
            now = utc_now_iso()
            self._execute(
                conn,
                "INSERT INTO tenants (id, name, plan, max_documents, max_memories, rate_limit_rpm, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'active', ?)",
                (safe_id, name, plan, max_documents, max_memories, rate_limit_rpm, now),
            )
            return self._fetchone(conn, "SELECT * FROM tenants WHERE id = ?", (safe_id,)) or {}

    def get_tenant(self, tenant_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            return self._fetchone(conn, "SELECT * FROM tenants WHERE id = ?", (tenant_id,))

    def list_tenants(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            return self._fetchall(conn, "SELECT * FROM tenants ORDER BY created_at DESC")

    # --- API Keys ---

    def create_api_key(self, tenant_id: str, name: str, role: str = "editor") -> dict[str, Any]:
        if role not in {"admin", "editor", "viewer"}:
            raise ValueError("role must be admin, editor, or viewer")
        if not self.get_tenant(tenant_id):
            raise FileNotFoundError(f"Tenant not found: {tenant_id}")
        raw_key = f"mlab_{secrets.token_urlsafe(24)}"
        key_id = uuid4().hex[:12]
        prefix = raw_key[:12]
        now = utc_now_iso()
        with self._conn() as conn:
            self._execute(
                conn,
                "INSERT INTO api_keys (id, tenant_id, name, role, key_prefix, key_hash, revoked, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
                (key_id, tenant_id, name, role, prefix, hash_secret(raw_key), now),
            )
        return {
            "id": key_id,
            "tenant_id": tenant_id,
            "name": name,
            "role": role,
            "key_prefix": prefix,
            "api_key": raw_key,  # Returned once on creation
            "created_at": now,
        }

    def lookup_api_key(self, raw_key: str) -> dict[str, Any] | None:
        digest = hash_secret(raw_key)
        with self._conn() as conn:
            row = self._fetchone(
                conn,
                "SELECT id, tenant_id, name, role, key_prefix, revoked, created_at FROM api_keys WHERE key_hash = ?",
                (digest,),
            )
            if not row or int(row["revoked"]) != 0:
                return None
            return row

    def list_api_keys(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
        with self._conn() as conn:
            if tenant_id:
                return self._fetchall(
                    conn,
                    "SELECT id, tenant_id, name, role, key_prefix, revoked, created_at FROM api_keys "
                    "WHERE tenant_id = ? ORDER BY created_at DESC",
                    (tenant_id,),
                )
            return self._fetchall(
                conn,
                "SELECT id, tenant_id, name, role, key_prefix, revoked, created_at FROM api_keys ORDER BY created_at DESC",
            )

    def revoke_api_key(self, key_id: str) -> bool:
        with self._conn() as conn:
            return self._execute(conn, "UPDATE api_keys SET revoked = 1 WHERE id = ?", (key_id,)) > 0

    # --- Async Jobs ---

    def create_job(self, tenant_id: str, job_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        job_id = uuid4().hex[:16]
        now = utc_now_iso()
        with self._conn() as conn:
            self._execute(
                conn,
                "INSERT INTO jobs (id, tenant_id, job_type, status, progress, payload_json, created_at, updated_at) "
                "VALUES (?, ?, ?, 'queued', 0, ?, ?, ?)",
                (job_id, tenant_id, job_type, json.dumps(payload, ensure_ascii=False), now, now),
            )
            return self._decode_job(self._fetchone(conn, "SELECT * FROM jobs WHERE id = ?", (job_id,)))

    def get_job(self, job_id: str, *, tenant_id: str | None = None) -> dict[str, Any] | None:
        with self._conn() as conn:
            if tenant_id:
                row = self._fetchone(conn, "SELECT * FROM jobs WHERE id = ? AND tenant_id = ?", (job_id, tenant_id))
            else:
                row = self._fetchone(conn, "SELECT * FROM jobs WHERE id = ?", (job_id,))
            return self._decode_job(row) if row else None

    def list_jobs(self, *, tenant_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as conn:
            if tenant_id:
                rows = self._fetchall(
                    conn,
                    "SELECT * FROM jobs WHERE tenant_id = ? ORDER BY created_at DESC LIMIT ?",
                    (tenant_id, limit),
                )
            else:
                rows = self._fetchall(conn, "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
            return [self._decode_job(r) for r in rows]

    def claim_next_queued_job(self, worker_id: str) -> dict[str, Any] | None:
        now = utc_now_iso()
        with self._conn() as conn:
            row = self._fetchone(conn, "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1")
            if not row:
                return None
            updated = self._execute(
                conn,
                "UPDATE jobs SET status = 'running', progress = 10, worker_id = ?, updated_at = ? "
                "WHERE id = ? AND status = 'queued'",
                (worker_id, now, row["id"]),
            )
            if updated == 0:
                return None
            return self._decode_job(self._fetchone(conn, "SELECT * FROM jobs WHERE id = ?", (row["id"],)))

    def update_job(
        self,
        job_id: str,
        *,
        status: str,
        progress: int,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        worker_id: str | None = None,
    ) -> dict[str, Any] | None:
        now = utc_now_iso()
        with self._conn() as conn:
            self._execute(
                conn,
                "UPDATE jobs SET status = ?, progress = ?, result_json = ?, error = ?, "
                "worker_id = COALESCE(?, worker_id), updated_at = ? WHERE id = ?",
                (
                    status,
                    progress,
                    json.dumps(result, ensure_ascii=False) if result is not None else None,
                    error,
                    worker_id,
                    now,
                    job_id,
                ),
            )
            row = self._fetchone(conn, "SELECT * FROM jobs WHERE id = ?", (job_id,))
            return self._decode_job(row) if row else None

    @staticmethod
    def _decode_job(row: dict[str, Any] | None) -> dict[str, Any]:
        assert row is not None
        return {
            "id": row["id"],
            "tenant_id": row["tenant_id"],
            "job_type": row["job_type"],
            "status": row["status"],
            "progress": int(row["progress"]),
            "payload": json.loads(row["payload_json"]),
            "result": json.loads(row["result_json"]) if row.get("result_json") else None,
            "error": row.get("error"),
            "worker_id": row.get("worker_id"),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # --- Audit Logs ---

    def record_audit(
        self,
        *,
        tenant_id: str,
        actor: str,
        role: str,
        action: str,
        resource: str,
        status_code: int,
        request_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._conn() as conn:
            self._execute(
                conn,
                "INSERT INTO audit_logs (id, tenant_id, actor, role, action, resource, status_code, request_id, metadata_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    uuid4().hex[:16],
                    tenant_id,
                    actor,
                    role,
                    action,
                    resource,
                    status_code,
                    request_id,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    utc_now_iso(),
                ),
            )

    def list_audit_logs(self, *, tenant_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self._conn() as conn:
            if tenant_id:
                rows = self._fetchall(
                    conn,
                    "SELECT * FROM audit_logs WHERE tenant_id = ? ORDER BY created_at DESC LIMIT ?",
                    (tenant_id, limit),
                )
            else:
                rows = self._fetchall(conn, "SELECT * FROM audit_logs ORDER BY created_at DESC LIMIT ?", (limit,))
            for r in rows:
                r["metadata"] = json.loads(r.pop("metadata_json"))
            return rows
