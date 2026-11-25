from uuid import uuid4
from memory_lab.core.models import InteractionTrace, utc_now_iso
from memory_lab.storage.filesystem.store import FileStore


class TraceStore:
    """Immutable append-only store for raw agent interaction traces."""

    def __init__(self, store: FileStore):
        self.store = store

    def append(
        self,
        *,
        content: str,
        session_id: str = "default",
        role: str = "user",
        timestamp: str | None = None,
        source_refs: list[str] | None = None,
        metadata: dict | None = None,
    ) -> InteractionTrace:
        safe_session = session_id.replace("/", "_").replace("..", "_")
        trace = InteractionTrace(
            id=uuid4().hex,
            session_id=safe_session,
            role=role,  # type: ignore[arg-type]
            content=content,
            timestamp=timestamp or utc_now_iso(),
            source_refs=list(source_refs or []),
            metadata=dict(metadata or {}),
        )
        self.store.put(f"memory/traces/{safe_session}/{trace.id}.json", trace.model_dump())
        return trace

    def get(self, trace_id: str) -> InteractionTrace:
        matches = list(self.store.root.glob(f"memory/traces/*/{trace_id}.json"))
        if not matches:
            raise FileNotFoundError(f"Trace not found: {trace_id}")
        return InteractionTrace.model_validate(self.store.get(str(matches[0].relative_to(self.store.root))))

    def list(self, *, session_id: str | None = None, limit: int = 100) -> list[InteractionTrace]:
        pattern = f"memory/traces/{session_id}/*.json" if session_id else "memory/traces/*/*.json"
        traces = [
            InteractionTrace.model_validate(self.store.get(str(path.relative_to(self.store.root))))
            for path in self.store.root.glob(pattern)
        ]
        traces.sort(key=lambda item: (item.timestamp, item.id), reverse=True)
        return traces[:limit]
