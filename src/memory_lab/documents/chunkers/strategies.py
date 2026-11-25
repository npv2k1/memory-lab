import logging
from typing import Any
from uuid import NAMESPACE_URL, uuid5
from memory_lab.core.models import Chunk, NormalizedDocument

logger = logging.getLogger(__name__)


def make_chunk(doc: NormalizedDocument, run_id: str, index: int, text: str, block_ids: list[str]) -> Chunk:
    return Chunk(
        id=str(uuid5(NAMESPACE_URL, f"{run_id}:{index}")),
        document_id=doc.document_id,
        version=doc.version,
        text=text,
        block_ids=block_ids,
        pipeline_run_id=run_id,
    )


def _snap_end(text: str, start: int, target_end: int) -> int:
    """Snap chunk end backward to the nearest paragraph, sentence, line, or word boundary."""
    if target_end >= len(text):
        return len(text)
    window = target_end - start
    min_pos = start + max(1, int(window * 0.65))

    # 1. Paragraph break
    idx = text.rfind("\n\n", min_pos, target_end)
    if idx != -1:
        return idx + 2

    # 2. Sentence break (. ! ? followed by whitespace)
    for sep in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
        idx = text.rfind(sep, min_pos, target_end)
        if idx != -1:
            return idx + len(sep)

    # 3. Line break
    idx = text.rfind("\n", min_pos, target_end)
    if idx != -1:
        return idx + 1

    # 4. Word boundary
    idx = text.rfind(" ", min_pos, target_end)
    if idx != -1:
        return idx + 1

    return target_end


def _snap_start(text: str, raw_start: int, prev_end: int) -> int:
    """Snap overlap start forward to a clean word/line boundary."""
    if raw_start <= 0:
        return 0
    if raw_start >= prev_end:
        return prev_end
    if text[raw_start - 1].isspace():
        return raw_start
    # Advance to next whitespace within a modest margin so we don't start mid-word
    limit = min(prev_end, raw_start + 40)
    for idx in range(raw_start, limit):
        if text[idx].isspace():
            nxt = idx + 1
            return nxt if nxt < prev_end else raw_start
    return raw_start


class FixedChunker:
    def __init__(self, size: int, overlap: int):
        if overlap >= size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.size, self.overlap = size, overlap

    def chunk(self, document: NormalizedDocument, artifact: dict, run_id: str) -> list[Chunk]:
        text, spans = "", []
        for block in document.blocks:
            start = len(text)
            text += block.text + "\n\n"
            spans.append((start, len(text), block.id))

        chunks: list[Chunk] = []
        pos = 0
        total = len(text)
        while pos < total:
            raw_end = min(pos + self.size, total)
            end = _snap_end(text, pos, raw_end)
            if end <= pos:
                end = raw_end
            segment = text[pos:end]
            if segment.strip():
                refs = [ref for left, right, ref in spans if left < end and right > pos]
                chunks.append(make_chunk(document, run_id, len(chunks), segment.strip(), refs))
            if end >= total:
                break
            next_raw = max(pos + 1, end - self.overlap)
            pos = _snap_start(text, next_raw, end)
        return chunks


def _extract_matching_refs(item: Any, known: set[str], visited: set[str] | None = None) -> list[str]:
    """Resolve item self_ref, its descendant refs, or its parent ref against known normalized block IDs."""
    visited = visited if visited is not None else set()
    ref = getattr(item, "self_ref", None)
    if ref and ref in known:
        return [ref]
    matched: list[str] = []
    for child in getattr(item, "children", None) or []:
        cref = getattr(child, "cref", None)
        if not cref or cref in visited:
            continue
        visited.add(cref)
        if cref in known:
            matched.append(cref)
    if matched:
        return matched
    parent = getattr(item, "parent", None)
    pref = getattr(parent, "cref", None) if parent is not None else None
    if pref and pref in known:
        return [pref]
    return []


class HybridChunkerAdapter:
    def __init__(self, chunk_size: int = 1200, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _build_chunker(self) -> Any:
        from docling.chunking import HybridChunker
        # Estimate token budget safely capped <= 512 to stay within typical embedding tokenizer limits
        max_tokens = max(64, min(512, self.chunk_size // 4))
        try:
            return HybridChunker(max_tokens=max_tokens)
        except TypeError:
            return HybridChunker()

    def chunk(self, document: NormalizedDocument, artifact: dict, run_id: str) -> list[Chunk]:
        from docling_core.types.doc import DoclingDocument
        if not artifact:
            raise ValueError("Hybrid chunking requires the Docling parser")
        chunker = self._build_chunker()
        parsed = DoclingDocument.model_validate(artifact)
        known = {block.id for block in document.blocks}
        result: list[Chunk] = []
        for chunk in chunker.chunk(dl_doc=parsed):
            refs: list[str] = []
            seen: set[str] = set()
            for item in getattr(chunk.meta, "doc_items", []) or []:
                for matched_ref in _extract_matching_refs(item, known):
                    if matched_ref not in seen:
                        seen.add(matched_ref)
                        refs.append(matched_ref)
            if not refs:
                # Fallback: preserve raw doc_item refs and warn instead of failing the entire run
                raw_refs = [
                    item.self_ref
                    for item in (getattr(chunk.meta, "doc_items", []) or [])
                    if getattr(item, "self_ref", None)
                ]
                logger.warning(
                    "hybrid_chunk_unmapped_blocks",
                    extra={"pipeline_run_id": run_id, "raw_refs": raw_refs},
                )
                refs = raw_refs or ([document.blocks[0].id] if document.blocks else [])
            text = chunker.contextualize(chunk)
            if text and text.strip():
                result.append(make_chunk(document, run_id, len(result), text, refs))
        return result
