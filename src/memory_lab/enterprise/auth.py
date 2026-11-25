import base64
from dataclasses import dataclass
import hashlib
import hmac
import json
import time
from typing import Any, Literal
from fastapi import HTTPException, Request
from memory_lab.core.config import Settings
from memory_lab.enterprise.store import EnterpriseStore

Role = Literal["viewer", "editor", "admin"]
ROLE_ORDER: dict[str, int] = {"viewer": 1, "editor": 2, "admin": 3}


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(encoded: str) -> bytes:
    padding = "=" * (-len(encoded) % 4)
    return base64.urlsafe_b64decode(encoded + padding)


def create_jwt(payload: dict[str, Any], secret: str, *, ttl_seconds: int = 3600) -> str:
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    body = {**payload, "iat": now, "exp": now + ttl_seconds}
    seg1 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    seg2 = _b64url_encode(json.dumps(body, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{seg1}.{seg2}".encode("ascii")
    sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{seg1}.{seg2}.{_b64url_encode(sig)}"


def verify_jwt(token: str, secret: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Malformed JWT")
    signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")
    expected_sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    provided_sig = _b64url_decode(parts[2])
    if not hmac.compare_digest(expected_sig, provided_sig):
        raise ValueError("Invalid JWT signature")
    payload = json.loads(_b64url_decode(parts[1]).decode("utf-8"))
    if int(payload.get("exp", 0)) < int(time.time()):
        raise ValueError("JWT expired")
    return payload


@dataclass(frozen=True)
class AuthContext:
    tenant_id: str
    actor: str
    role: Role
    auth_method: str

    def has_role(self, required: Role) -> bool:
        return ROLE_ORDER.get(self.role, 0) >= ROLE_ORDER.get(required, 99)


class AuthService:
    """Authenticates requests via API Key, Bearer HS256 JWT, or Bootstrap Admin Secret with RBAC."""

    def __init__(self, settings: Settings, store: EnterpriseStore):
        self.settings = settings
        self.store = store

    def authenticate(self, request: Request) -> AuthContext:
        api_key = request.headers.get("X-API-Key", "").strip()
        auth_header = request.headers.get("Authorization", "").strip()
        admin_secret = request.headers.get("X-Admin-Secret", "").strip()
        requested_tenant = request.headers.get("X-Tenant-ID", "").strip().lower()

        # 1. Bootstrap admin secret (platform operator)
        if self.settings.admin_bootstrap_key and admin_secret:
            if hmac.compare_digest(admin_secret, self.settings.admin_bootstrap_key):
                tenant = requested_tenant or self.settings.default_tenant_id
                self.store.ensure_tenant(tenant, tenant)
                return AuthContext(tenant_id=tenant, actor="platform-admin", role="admin", auth_method="admin_secret")
            raise HTTPException(status_code=401, detail="Invalid admin secret")

        # 2. API Key authentication
        if api_key:
            key_row = self.store.lookup_api_key(api_key)
            if not key_row:
                raise HTTPException(status_code=401, detail="Invalid or revoked API key")
            tenant = key_row["tenant_id"]
            if requested_tenant and requested_tenant != tenant and key_row["role"] != "admin":
                raise HTTPException(status_code=403, detail="Cross-tenant access denied")
            target_tenant = requested_tenant if (requested_tenant and key_row["role"] == "admin") else tenant
            return AuthContext(
                tenant_id=target_tenant,
                actor=f"key:{key_row['id']}:{key_row['name']}",
                role=key_row["role"],  # type: ignore[arg-type]
                auth_method="api_key",
            )

        # 3. Bearer HS256 JWT authentication
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
            try:
                claims = verify_jwt(token, self.settings.jwt_secret)
            except Exception as exc:
                raise HTTPException(status_code=401, detail=f"Invalid bearer token: {exc}") from exc
            tenant = str(claims.get("tenant_id", self.settings.default_tenant_id))
            role = str(claims.get("role", "viewer"))
            if role not in ROLE_ORDER:
                raise HTTPException(status_code=401, detail="Invalid token role")
            if requested_tenant and requested_tenant != tenant and role != "admin":
                raise HTTPException(status_code=403, detail="Cross-tenant access denied")
            target_tenant = requested_tenant if (requested_tenant and role == "admin") else tenant
            return AuthContext(
                tenant_id=target_tenant,
                actor=str(claims.get("sub", "jwt-user")),
                role=role,  # type: ignore[arg-type]
                auth_method="jwt",
            )

        # 4. Open mode (local dev / single-tenant / E2E backwards compatibility)
        if self.settings.auth_mode == "disabled":
            tenant = requested_tenant or self.settings.default_tenant_id
            if tenant != self.settings.default_tenant_id:
                self.store.ensure_tenant(tenant, tenant)
            return AuthContext(tenant_id=tenant, actor="local-dev", role="admin", auth_method="disabled")

        raise HTTPException(status_code=401, detail="Authentication required (X-API-Key or Authorization: Bearer)")

    def require_role(self, ctx: AuthContext, required: Role) -> None:
        if not ctx.has_role(required):
            raise HTTPException(status_code=403, detail=f"Insufficient permissions: requires '{required}' role")

    def issue_token_for_context(self, ctx: AuthContext) -> dict[str, Any]:
        token = create_jwt(
            {"sub": ctx.actor, "tenant_id": ctx.tenant_id, "role": ctx.role},
            self.settings.jwt_secret,
            ttl_seconds=self.settings.jwt_ttl_seconds,
        )
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": self.settings.jwt_ttl_seconds,
            "tenant_id": ctx.tenant_id,
            "role": ctx.role,
        }
