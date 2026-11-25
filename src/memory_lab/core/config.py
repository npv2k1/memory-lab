from pathlib import Path
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
