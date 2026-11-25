from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class Block(BaseModel):
    id: str
    text: str
    type: str = "text"
    heading_hierarchy: list[str] = Field(default_factory=list)
    page: int | None = None
    bbox: dict[str, Any] | None = None


class ParsedDocument(BaseModel):
    blocks: list[Block]
    artifact: dict[str, Any] = Field(default_factory=dict)


class NormalizedDocument(BaseModel):
    document_id: str
    version: str
    blocks: list[Block]


class Document(BaseModel):
    id: str
    version: str
    filename: str
    content_hash: str


class Chunk(BaseModel):
    id: str
    document_id: str
    version: str
    text: str
    block_ids: list[str]
    pipeline_run_id: str


class PipelineConfig(BaseModel):
    parser: Literal["docling", "text"] = "docling"
    chunker: Literal["fixed", "hybrid"] = "fixed"
    chunk_size: int = Field(default=1200, ge=100, le=10000)
    overlap: int = Field(default=150, ge=0)
    enrich: bool = True


class Evidence(BaseModel):
    chunk: Chunk
    score: float
    source_url: str
    score_breakdown: dict[str, float] = Field(default_factory=dict)


# --- Phase 3: Non-generative Enrichment & Temporal Models ---

class Entity(BaseModel):
    name: str
    normalized_name: str
    type: Literal["person", "organization", "technology", "location", "date", "concept", "other"] = "concept"


class Relation(BaseModel):
    subject: str
    predicate: str
    object: str
    evidence_id: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class TemporalAnchor(BaseModel):
    raw: str
    normalized: str
    granularity: Literal["year", "month", "day", "relative"] = "day"


class EnrichedChunk(BaseModel):
    chunk_id: str
    document_id: str
    version: str
    block_ids: list[str] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    temporal: list[TemporalAnchor] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class EnrichmentArtifact(BaseModel):
    pipeline_run_id: str
    document_id: str
    version: str
    chunks: list[EnrichedChunk] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    temporal: list[TemporalAnchor] = Field(default_factory=list)


# --- Phase 2, 3 & ai-memory Integration Models ---

MemoryStrategyName = Literal["vector", "graph", "zero_token", "hybrid"]
MemoryTierName = Literal["working", "episodic", "semantic", "procedural"]
FeedbackSignalName = Literal["helpful", "not_helpful", "stale", "wrong"]
EpistemicNature = Literal["measured", "reported", "inferred", "unknown"]
WikiPageKind = Literal["concept", "decision", "gotcha", "procedure", "rule", "note"]
PipelineStageName = Literal["parse", "normalize", "chunk", "enrich", "index"]
SearchMode = Literal["dense", "sparse", "hybrid"]


class InteractionTrace(BaseModel):
    id: str
    session_id: str = "default"
    role: Literal["user", "assistant", "system", "tool"] = "user"
    content: str
    timestamp: str = Field(default_factory=utc_now_iso)
    source_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryRecord(BaseModel):
    id: str
    strategy: MemoryStrategyName
    namespace: str = "default"
    project: str = "default"
    tier: MemoryTierName = "episodic"
    pinned: bool = False
    salience: float = Field(default=1.0, ge=0.25, le=2.0)
    access_count: int = Field(default=0, ge=0)
    last_accessed_at: str | None = None
    expires_at: str | None = None
    archived: bool = False
    archived_reason: str | None = None
    feedback_history: list[dict[str, Any]] = Field(default_factory=list)
    content: str
    summary: str = ""
    trace_ids: list[str] = Field(default_factory=list)
    session_id: str = "default"
    role: Literal["user", "assistant", "system", "tool"] = "user"
    timestamp: str = Field(default_factory=utc_now_iso)
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    temporal: list[TemporalAnchor] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    hierarchy_level: Literal["raw", "session", "day", "month", "year"] = "raw"
    tags: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    deleted: bool = False


class MemorySearchHit(BaseModel):
    record: MemoryRecord
    score: float
    strategy: str
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    evidence_trace_ids: list[str] = Field(default_factory=list)


class MemoryStoreRequest(BaseModel):
    content: str = Field(min_length=1, max_length=50000)
    session_id: str = Field(default="default", min_length=1, max_length=128)
    project: str = Field(default="default", min_length=1, max_length=128)
    tier: MemoryTierName = "episodic"
    pinned: bool = False
    expires_at: str | None = None
    scrub_secrets: bool = True
    role: Literal["user", "assistant", "system", "tool"] = "user"
    strategy: MemoryStrategyName | None = None
    timestamp: str | None = None
    tags: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10000)
    strategy: MemoryStrategyName | None = None
    limit: int = Field(default=5, ge=1, le=50)
    session_id: str | None = None
    project: str | None = None
    temporal_bucket: str | None = None


class MemoryFeedbackRequest(BaseModel):
    signal: FeedbackSignalName
    reason: str = Field(default="", max_length=500)


class ForgetSweepRequest(BaseModel):
    dry_run: bool = True
    project: str | None = None
    cold_threshold: float = Field(default=0.20, ge=0.0, le=2.0)
    lambda_decay: float = Field(default=0.02, ge=0.0, le=5.0)


class HandoffCreateRequest(BaseModel):
    summary: str = Field(min_length=1, max_length=10000)
    project: str = Field(default="default", min_length=1, max_length=128)
    session_id: str = Field(default="default", min_length=1, max_length=128)
    from_agent: str = Field(default="agent", min_length=1, max_length=128)
    next_steps: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    files_touched: list[str] = Field(default_factory=list)
    shared: bool = True


class HandoffAcceptRequest(BaseModel):
    handoff_id: str | None = None
    project: str | None = None
    agent: str = Field(default="agent", min_length=1, max_length=128)


class HandoffRecord(BaseModel):
    id: str
    project: str = "default"
    session_id: str = "default"
    from_agent: str = "agent"
    summary: str
    next_steps: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    files_touched: list[str] = Field(default_factory=list)
    shared: bool = True
    status: Literal["pending", "accepted", "cancelled"] = "pending"
    created_at: str = Field(default_factory=utc_now_iso)
    accepted_at: str | None = None
    accepted_by: str | None = None
    cancelled_at: str | None = None


class AgentMessageSendRequest(BaseModel):
    to_project: str = Field(min_length=1, max_length=128)
    body: str = Field(min_length=1, max_length=20000)
    from_project: str = Field(default="default", min_length=1, max_length=128)
    from_agent: str = Field(default="agent", min_length=1, max_length=128)
    to_agent: str | None = Field(default=None, max_length=128)
    subject: str = Field(default="", max_length=300)
    nature: EpistemicNature = "unknown"
    evidence: str = Field(default="", max_length=4000)


class AgentMessagePopRequest(BaseModel):
    project: str = Field(default="default", min_length=1, max_length=128)
    message_id: str | None = None
    agent: str | None = None


class AgentMessageCancelRequest(BaseModel):
    project: str = Field(default="default", min_length=1, max_length=128)
    message_id: str | None = None


class AgentMessageRecord(BaseModel):
    id: str
    from_project: str
    to_project: str
    from_agent: str = "agent"
    to_agent: str | None = None
    subject: str = ""
    body: str
    nature: EpistemicNature = "unknown"
    evidence: str = ""
    status: Literal["pending", "claimed", "cancelled"] = "pending"
    created_at: str = Field(default_factory=utc_now_iso)
    claimed_at: str | None = None
    claimed_by: str | None = None
    cancelled_at: str | None = None


class WikiPageWriteRequest(BaseModel):
    path: str = Field(min_length=1, max_length=256)
    title: str = Field(min_length=1, max_length=300)
    content: str = Field(min_length=1, max_length=100000)
    kind: WikiPageKind = "note"
    tier: MemoryTierName = "semantic"
    project: str = Field(default="default", min_length=1, max_length=128)
    pinned: bool = False
    tags: list[str] = Field(default_factory=list)
    expires_at: str | None = None
    author: str = Field(default="agent", max_length=128)


class WikiPagePatchRequest(BaseModel):
    path: str = Field(min_length=1, max_length=256)
    append: str | None = None
    replace_old: str | None = None
    replace_new: str | None = None
    title: str | None = None
    pinned: bool | None = None
    tags: list[str] | None = None
    author: str = Field(default="agent", max_length=128)


class WikiPageRestoreRequest(BaseModel):
    path: str = Field(min_length=1, max_length=256)
    version: int | None = Field(default=None, ge=1)


class WikiPageRecord(BaseModel):
    path: str
    title: str
    content: str
    kind: WikiPageKind = "note"
    tier: MemoryTierName = "semantic"
    project: str = "default"
    pinned: bool = False
    salience: float = Field(default=1.0, ge=0.25, le=2.0)
    access_count: int = Field(default=0, ge=0)
    last_accessed_at: str | None = None
    tags: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)
    version: int = 1
    deleted: bool = False
    author: str = "agent"
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    expires_at: str | None = None


class TemporalSummaryNode(BaseModel):
    bucket: str
    granularity: Literal["year", "month", "day", "session"]
    record_ids: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    record_count: int = 0
    children: list[str] = Field(default_factory=list)


# --- Phase 4: Experimentation & Stage Invalidation Models ---

class StageInvalidationRequest(BaseModel):
    stages: list[PipelineStageName] = Field(min_length=1)


class BenchmarkCase(BaseModel):
    query: str = Field(min_length=1, max_length=10000)
    relevant_document_ids: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_block_ids: list[str] = Field(default_factory=list)
    relevant_memory_ids: list[str] = Field(default_factory=list)
    relevant_substrings: list[str] = Field(default_factory=list)


class BenchmarkDatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    cases: list[BenchmarkCase] = Field(min_length=1)


class BenchmarkDataset(BaseModel):
    id: str
    name: str
    description: str = ""
    dataset_hash: str
    cases: list[BenchmarkCase]
    created_at: str = Field(default_factory=utc_now_iso)


class ExperimentRunRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    dataset_id: str = Field(min_length=1)
    target: Literal["documents", "memory"] = "documents"
    k: int = Field(default=5, ge=1, le=50)
    search_mode: SearchMode = "dense"
    memory_strategy: MemoryStrategyName | None = None
    document_id: str | None = None
    version: str | None = None
    pipeline_run_id: str | None = None
    session_id: str | None = None
    temporal_bucket: str | None = None
    notes: str = ""


class ExperimentCaseResult(BaseModel):
    query: str
    retrieved_ids: list[str] = Field(default_factory=list)
    matched_ranks: list[int] = Field(default_factory=list)
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    latency_ms: float


class ExperimentReport(BaseModel):
    id: str
    name: str
    dataset_id: str
    dataset_hash: str
    target: Literal["documents", "memory"]
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)
    k: int
    metrics: dict[str, float] = Field(default_factory=dict)
    case_results: list[ExperimentCaseResult] = Field(default_factory=list)
