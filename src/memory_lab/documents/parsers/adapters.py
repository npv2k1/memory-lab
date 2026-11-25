from importlib.metadata import version
from pathlib import Path
import threading
from typing import Any
from memory_lab.core.models import Block, ParsedDocument

_CONVERTER_LOCK = threading.Lock()
_CACHED_CONVERTER: Any = None


def _get_docling_converter() -> Any:
    global _CACHED_CONVERTER
    with _CONVERTER_LOCK:
        if _CACHED_CONVERTER is None:
            from docling.document_converter import DocumentConverter
            _CACHED_CONVERTER = DocumentConverter()
        return _CACHED_CONVERTER


def _resolve_docling_ref(parsed: Any, cref: str) -> Any | None:
    if hasattr(parsed, "resolve"):
        try:
            return parsed.resolve(cref)
        except Exception:
            pass
    # Fallback manual resolution for #/texts/N, #/groups/N, #/tables/N
    if isinstance(cref, str) and cref.startswith("#/"):
        parts = cref[2:].split("/")
        if len(parts) == 2 and parts[1].isdigit():
            collection = getattr(parsed, parts[0], None)
            idx = int(parts[1])
            if isinstance(collection, list) and 0 <= idx < len(collection):
                return collection[idx]
    return None


def _collect_children_text(parsed: Any, item: Any, visited: set[str]) -> tuple[str, set[str]]:
    parts: list[str] = []
    consumed: set[str] = set()
    for child_ref in getattr(item, "children", None) or []:
        cref = getattr(child_ref, "cref", None)
        if not cref or cref in visited:
            continue
        visited.add(cref)
        child = _resolve_docling_ref(parsed, cref)
        if child is None:
            continue
        child_kind = str(getattr(child, "label", "text")).lower()
        # Only fold inline/text/code child nodes into parent container; keep nested tables/headings separate
        if "table" in child_kind or "heading" in child_kind or "title" in child_kind:
            continue
        child_text = (getattr(child, "text", "") or "").strip()
        if child_text:
            parts.append(child_text)
            consumed.add(cref)
        nested_text, nested_consumed = _collect_children_text(parsed, child, visited)
        if nested_text:
            parts.append(nested_text)
            consumed.add(cref)
            consumed.update(nested_consumed)
    return " ".join(parts).strip(), consumed


class TextParser:
    version = "1"

    def parse(self, path: Path) -> ParsedDocument:
        if path.suffix not in {".txt", ".md"}:
            raise ValueError("Text parser only supports UTF-8 TXT/Markdown")
        blocks = [
            Block(id=f"text:{i}", text=text)
            for i, text in enumerate(path.read_text(encoding="utf-8").split("\n\n"))
            if text.strip()
        ]
        return ParsedDocument(blocks=blocks)


class DoclingParser:
    @property
    def version(self) -> str:
        return version("docling")

    def parse(self, path: Path) -> ParsedDocument:
        parsed = _get_docling_converter().convert(path).document
        blocks: list[Block] = []
        headings: list[str] = []
        consumed_inline_refs: set[str] = set()

        for item, level in parsed.iterate_items():
            ref = getattr(item, "self_ref", "")
            if ref and ref in consumed_inline_refs:
                continue

            kind = str(getattr(item, "label", "text"))
            text = (getattr(item, "text", "") or "").strip()
            if "table" in kind.lower():
                text = item.export_to_markdown(doc=parsed)
            else:
                # Merge inline children (e.g., inline code inside Markdown list_item/paragraph)
                # so parent blocks are never dropped and inline tokens are not fragmented into separate blocks.
                child_text, child_refs = _collect_children_text(parsed, item, {ref} if ref else set())
                if child_text:
                    text = f"{text} {child_text}".strip() if text else child_text
                    consumed_inline_refs.update(child_refs)

            if not text:
                continue
            if "heading" in kind.lower() or "title" in kind.lower():
                headings = headings[: max(level - 1, 0)] + [text]
            provenance = item.prov[0] if getattr(item, "prov", []) else None
            blocks.append(
                Block(
                    id=ref,
                    text=text,
                    type=kind,
                    heading_hierarchy=headings.copy(),
                    page=provenance.page_no if provenance else None,
                    bbox=provenance.bbox.model_dump(mode="json") if provenance else None,
                )
            )
        return ParsedDocument(blocks=blocks, artifact=parsed.export_to_dict())
