from typing import Any, Literal
from pydantic import BaseModel, Field


class Block(BaseModel):
    id: str
    text: str
    type: str = "text"
    heading_hierarchy: list[str] = Field(default_factory=list)
    page: int | None = None
    bbox: dict[str, Any] | None = None


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


class Evidence(BaseModel):
    chunk: Chunk
    score: float
    source_url: str
