import time
from typing import Any
from urllib.parse import urlparse
from pydantic import BaseModel, Field
from memory_lab.core.config import Settings
from memory_lab.core.models import utc_now_iso
from memory_lab.documents.pipeline.service import DocumentPipeline
from memory_lab.memory.service import MemoryService
from memory_lab.retrieval.embedding import OpenAIEmbeddings
from memory_lab.storage.filesystem.store import FileStore

EMBEDDING_PRESETS: list[dict[str, Any]] = [
    {
        "id": "ollama-bge-m3",
        "label": "Ollama · bge-m3 (Multilingual VI/EN)",
        "provider": "ollama",
        "base_url": "http://host.docker.internal:11434/v1",
        "model": "bge-m3",
        "dimensions": 1024,
        "multilingual": True,
        "notes": "Khuyên dùng cho tiếng Việt + tiếng Anh (cross-lingual). Chạy `ollama pull bge-m3`.",
    },
    {
        "id": "ollama-multilingual-e5",
        "label": "Ollama · multilingual-e5-large",
        "provider": "ollama",
        "base_url": "http://host.docker.internal:11434/v1",
        "model": "jeffh/intfloat-multilingual-e5-large:f16",
        "dimensions": 1024,
        "multilingual": True,
        "notes": "Chất lượng cao cho tìm kiếm đa ngôn ngữ VI/EN.",
    },
    {
        "id": "ollama-nomic",
        "label": "Ollama · nomic-embed-text (Fast Local EN)",
        "provider": "ollama",
        "base_url": "http://host.docker.internal:11434/v1",
        "model": "nomic-embed-text",
        "dimensions": 768,
        "multilingual": False,
        "notes": "Nhẹ, nhanh (~15ms), tối ưu cho tiếng Anh.",
    },
    {
        "id": "openai-small",
        "label": "OpenAI · text-embedding-3-small",
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "model": "text-embedding-3-small",
        "dimensions": 1536,
        "multilingual": True,
        "notes": "Cloud API của OpenAI, hỗ trợ tốt VI/EN. Yêu cầu API Key.",
    },
    {
        "id": "openai-large",
        "label": "OpenAI · text-embedding-3-large",
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "model": "text-embedding-3-large",
        "dimensions": 3072,
        "multilingual": True,
        "notes": "Độ chính xác cao nhất của OpenAI (3072 dims). Yêu cầu API Key.",
    },
    {
        "id": "lmstudio",
        "label": "LM Studio · Local OpenAI-Compatible",
        "provider": "lmstudio",
        "base_url": "http://host.docker.internal:1234/v1",
        "model": "text-embedding-bge-m3",
        "dimensions": 1024,
        "multilingual": True,
        "notes": "Kết nối LM Studio local server.",
    },
    {
        "id": "custom",
        "label": "Custom OpenAI-Compatible Endpoint",
        "provider": "custom",
        "base_url": "",
        "model": "",
        "dimensions": None,
        "multilingual": True,
        "notes": "Hỗ trợ bất kỳ provider chuẩn `/v1/embeddings` nào (vLLM, LiteLLM, Azure, Together, SiliconFlow...).",
    },
]


class EmbeddingSettingsUpdate(BaseModel):
    provider: str = Field(default="custom", min_length=1, max_length=64)
    base_url: str = Field(min_length=1, max_length=512)
    model: str = Field(min_length=1, max_length=256)
    api_key: str | None = Field(default=None, max_length=512)
    clear_api_key: bool = False


class EmbeddingTestRequest(BaseModel):
    base_url: str | None = Field(default=None, max_length=512)
    model: str | None = Field(default=None, max_length=256)
    api_key: str | None = Field(default=None, max_length=512)
    sample_text: str = Field(default="Memory Lab embedding connectivity test", min_length=1, max_length=2000)


def _mask_api_key(api_key: str) -> str:
    if not api_key:
        return ""
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return f"{api_key[:3]}***{api_key[-4:]}"


def _validate_base_url(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be a valid http:// or https:// URL")
    return cleaned


def _infer_provider(base_url: str) -> str:
    lower = base_url.lower()
    if "openai.com" in lower:
        return "openai"
    if ":11434" in lower or "ollama" in lower:
        return "ollama"
    if ":1234" in lower:
        return "lmstudio"
    return "custom"


class EmbeddingSettingsService:
    """Manages per-tenant runtime Embedding Provider settings with persistence and live hot-swap."""

    SETTINGS_KEY = "settings/embedding.json"

    def _load_persisted(self, store: FileStore) -> dict[str, Any] | None:
        if not store.path(self.SETTINGS_KEY).exists():
            return None
        try:
            data = store.get(self.SETTINGS_KEY)
            if isinstance(data, dict) and data.get("base_url") and data.get("model"):
                return data
        except Exception:
            return None
        return None

    def _resolve_raw(self, store: FileStore, default_settings: Settings, pipeline: DocumentPipeline) -> dict[str, Any]:
        persisted = self._load_persisted(store)
        if persisted:
            return {
                "provider": persisted.get("provider") or _infer_provider(persisted["base_url"]),
                "base_url": str(persisted["base_url"]).rstrip("/"),
                "model": str(persisted["model"]),
                "api_key": str(persisted.get("api_key") or ""),
                "source": "persisted",
                "updated_at": persisted.get("updated_at"),
            }
        emb = pipeline.embedding
        base_url = getattr(emb, "base_url", None) or default_settings.embedding_base_url
        model = getattr(emb, "model", None) or default_settings.embedding_model
        api_key = getattr(emb, "api_key", None)
        if api_key is None:
            api_key = default_settings.embedding_api_key
        return {
            "provider": _infer_provider(str(base_url)),
            "base_url": str(base_url).rstrip("/"),
            "model": str(model),
            "api_key": str(api_key or ""),
            "source": "env",
            "updated_at": None,
        }

    def get_public(
        self,
        store: FileStore,
        default_settings: Settings,
        pipeline: DocumentPipeline,
        memory: MemoryService,
    ) -> dict[str, Any]:
        raw = self._resolve_raw(store, default_settings, pipeline)
        api_key = raw["api_key"]
        return {
            "provider": raw["provider"],
            "base_url": raw["base_url"],
            "model": raw["model"],
            "has_api_key": bool(api_key),
            "api_key_masked": _mask_api_key(api_key),
            "embedding_identity": pipeline.embedding_identity,
            "document_namespace": pipeline.namespace,
            "memory_namespace": memory.vector.memory_namespace,
            "source": raw["source"],
            "updated_at": raw["updated_at"],
            "presets": EMBEDDING_PRESETS,
        }

    def apply_persisted_if_exists(
        self,
        store: FileStore,
        tenant_id: str,
        default_tenant_id: str,
        pipeline: DocumentPipeline,
        memory: MemoryService,
    ) -> bool:
        persisted = self._load_persisted(store)
        if not persisted:
            return False
        base_url = str(persisted["base_url"]).rstrip("/")
        model = str(persisted["model"])
        api_key = str(persisted.get("api_key") or "")
        identity = f"{base_url}:{model}" if tenant_id == default_tenant_id else f"{tenant_id}:{base_url}:{model}"
        provider = OpenAIEmbeddings(base_url, api_key, model)
        pipeline.configure_embedding(provider, identity)
        memory.configure_embedding(provider, identity)
        return True

    def update(
        self,
        store: FileStore,
        tenant_id: str,
        default_tenant_id: str,
        payload: EmbeddingSettingsUpdate,
        pipeline: DocumentPipeline,
        memory: MemoryService,
        default_settings: Settings,
    ) -> dict[str, Any]:
        base_url = _validate_base_url(payload.base_url)
        model = payload.model.strip()
        if not model:
            raise ValueError("model must not be empty")

        current = self._resolve_raw(store, default_settings, pipeline)
        if payload.clear_api_key:
            resolved_api_key = ""
        elif payload.api_key is not None and payload.api_key != "":
            resolved_api_key = payload.api_key.strip()
        else:
            resolved_api_key = current["api_key"]

        prev_namespace = pipeline.namespace
        record = {
            "provider": payload.provider.strip() or _infer_provider(base_url),
            "base_url": base_url,
            "model": model,
            "api_key": resolved_api_key,
            "updated_at": utc_now_iso(),
        }
        with store.lock("embedding_settings"):
            store.put(self.SETTINGS_KEY, record)

        identity = f"{base_url}:{model}" if tenant_id == default_tenant_id else f"{tenant_id}:{base_url}:{model}"
        provider = OpenAIEmbeddings(base_url, resolved_api_key, model)
        pipeline.configure_embedding(provider, identity)
        memory.configure_embedding(provider, identity)

        public = self.get_public(store, default_settings, pipeline, memory)
        public["namespace_changed"] = prev_namespace != pipeline.namespace
        public["previous_namespace"] = prev_namespace
        return public

    def test_connection(
        self,
        store: FileStore,
        payload: EmbeddingTestRequest,
        pipeline: DocumentPipeline,
        default_settings: Settings,
    ) -> dict[str, Any]:
        current = self._resolve_raw(store, default_settings, pipeline)
        use_override = bool(payload.base_url or payload.model or payload.api_key)
        base_url = _validate_base_url(payload.base_url) if payload.base_url else current["base_url"]
        model = payload.model.strip() if payload.model else current["model"]
        api_key = payload.api_key.strip() if (payload.api_key is not None and payload.api_key != "") else current["api_key"]

        started = time.perf_counter()
        try:
            if not use_override and not isinstance(pipeline.embedding, OpenAIEmbeddings):
                vectors = pipeline.embedding.embed([payload.sample_text])
            else:
                if not base_url:
                    raise ValueError("base_url is required")
                if not model:
                    raise ValueError("model is required")
                client = OpenAIEmbeddings(base_url, api_key, model)
                vectors = client.embed([payload.sample_text])
            latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
            dim = len(vectors[0]) if vectors and vectors[0] else 0
            return {
                "ok": True,
                "dimension": dim,
                "latency_ms": latency_ms,
                "base_url": base_url,
                "model": model,
                "error": None,
            }
        except Exception as exc:
            latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
            return {
                "ok": False,
                "dimension": 0,
                "latency_ms": latency_ms,
                "base_url": base_url,
                "model": model,
                "error": str(exc),
            }
