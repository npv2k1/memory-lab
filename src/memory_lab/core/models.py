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


# --- Phase 2 & 3: Memory Models ---

MemoryStrategyName = Literal["vector", "graph", "zero_token", "hybrid"]
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
    temporal_bucket: str | None = None


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
