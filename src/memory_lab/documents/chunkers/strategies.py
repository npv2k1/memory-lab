from uuid import NAMESPACE_URL, uuid5
from memory_lab.core.models import Chunk, NormalizedDocument


def make_chunk(doc: NormalizedDocument, run_id: str, index: int, text: str, block_ids: list[str]) -> Chunk:
    return Chunk(id=str(uuid5(NAMESPACE_URL, f"{run_id}:{index}")), document_id=doc.document_id,
        version=doc.version, text=text, block_ids=block_ids, pipeline_run_id=run_id)


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
        chunks = []
        for start in range(0, len(text), self.size - self.overlap):
            end = min(start + self.size, len(text))
            if text[start:end].strip():
                refs = [ref for left, right, ref in spans if left < end and right > start]
                chunks.append(make_chunk(document, run_id, len(chunks), text[start:end], refs))
            if end == len(text):
                break
        return chunks


class HybridChunkerAdapter:
    def chunk(self, document: NormalizedDocument, artifact: dict, run_id: str) -> list[Chunk]:
        from docling.chunking import HybridChunker
        from docling_core.types.doc import DoclingDocument
        if not artifact:
            raise ValueError("Hybrid chunking requires the Docling parser")
        chunker = HybridChunker()
        parsed = DoclingDocument.model_validate(artifact)
        known = {block.id for block in document.blocks}
        result = []
        for chunk in chunker.chunk(dl_doc=parsed):
            refs = [item.self_ref for item in chunk.meta.doc_items if item.self_ref in known]
            if not refs:
                raise ValueError("Hybrid chunk has no normalized source blocks")
            result.append(make_chunk(document, run_id, len(result), chunker.contextualize(chunk), refs))
        return result
