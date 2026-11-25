from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    data_dir: Path = Path("data")
    embedding_base_url: str = "http://localhost:1234/v1"
    embedding_api_key: str = ""
    embedding_model: str = ""
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    neo4j_url: str = ""
    neo4j_user: str = "neo4j"
    neo4j_password: str = "memory-lab-local"
    neo4j_database: str = "neo4j"
    default_memory_strategy: str = "hybrid"
    cors_origins: list[str] = ["http://localhost:4321"]
    max_upload_bytes: int = 50 * 1024 * 1024

    # --- Enterprise SaaS Multi-Tenancy, Auth, Queue, Object Storage & Observability ---
    auth_mode: Literal["disabled", "enterprise"] = "disabled"
    jwt_secret: str = "change-me-enterprise-jwt-secret"
    jwt_ttl_seconds: int = 3600
    admin_bootstrap_key: str = ""
    default_tenant_id: str = "default"
    database_url: str = ""
    redis_url: str = ""
    rate_limit_rpm: int = 0  # 0 = disabled by default; set e.g. 600 in enterprise prod
    s3_endpoint_url: str = ""
    s3_bucket: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_region: str = "us-east-1"
    metrics_enabled: bool = True
