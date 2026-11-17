from importlib.metadata import version
from pathlib import Path
from memory_lab.core.models import Block, ParsedDocument


class TextParser:
    version = "1"

    def parse(self, path: Path) -> ParsedDocument:
        if path.suffix not in {".txt", ".md"}:
            raise ValueError("Text parser only supports UTF-8 TXT/Markdown")
        blocks = [Block(id=f"text:{i}", text=text) for i, text in enumerate(path.read_text(encoding="utf-8").split("\n\n")) if text.strip()]
        return ParsedDocument(blocks=blocks)


class DoclingParser:
    @property
    def version(self) -> str:
        return version("docling")

    def parse(self, path: Path) -> ParsedDocument:
        from docling.document_converter import DocumentConverter
        parsed = DocumentConverter().convert(path).document
        blocks, headings = [], []
        for item, level in parsed.iterate_items():
            text = getattr(item, "text", "")
            kind = str(getattr(item, "label", "text"))
            if "table" in kind:
                text = item.export_to_markdown(doc=parsed)
            if not text:
                continue
            if "heading" in kind or "title" in kind:
                headings = headings[:max(level - 1, 0)] + [text]
            provenance = item.prov[0] if getattr(item, "prov", []) else None
            blocks.append(Block(id=item.self_ref, text=text, type=kind,
                heading_hierarchy=headings.copy(), page=provenance.page_no if provenance else None,
                bbox=provenance.bbox.model_dump(mode="json") if provenance else None))
        return ParsedDocument(blocks=blocks, artifact=parsed.export_to_dict())
