import json
import re
from typing import Any
from memory_lab.core.models import (
    WikiPageKind,
    WikiPagePatchRequest,
    WikiPageRecord,
    WikiPageWriteRequest,
    utc_now_iso,
)
from memory_lab.documents.enrichment.extractor import NonGenerativeExtractor
from memory_lab.memory.scrub import scrub_secrets
from memory_lab.retrieval.sparse import BM25Scorer
from memory_lab.storage.filesystem.store import FileStore, digest

WIKILINK_RE = re.compile(r"\[\[([^\[\]|#]+)(?:[#|][^\[\]]+)?\]\]")

KIND_AUTHORITY: dict[str, float] = {
    "rule": 1.30,
    "procedure": 1.20,
    "decision": 1.20,
    "gotcha": 1.15,
    "concept": 1.05,
    "note": 1.00,
}


def normalize_wiki_path(raw_path: str) -> str:
    """Validate and normalize a relative wiki path (blocks traversal, enforces .md suffix)."""
    cleaned = raw_path.strip().replace("\\", "/").lstrip("/")
    if not cleaned or "\x00" in cleaned:
        raise ValueError("Invalid wiki page path")
    parts = [p for p in cleaned.split("/") if p]
    if any(p in {".", ".."} for p in parts):
        raise ValueError("Wiki page path must not contain '.' or '..'")
    if any(not re.fullmatch(r"[A-Za-z0-9._-]+", p) for p in parts):
        raise ValueError("Wiki page path segments may only contain letters, digits, '.', '_' and '-'")
    normalized = "/".join(parts)
    if not normalized.endswith(".md"):
        normalized += ".md"
    return normalized


def _infer_kind_from_path(path: str, explicit_kind: WikiPageKind) -> WikiPageKind:
    if explicit_kind != "note":
        return explicit_kind
    prefix = path.split("/", 1)[0].lower() if "/" in path else ""
    mapping: dict[str, WikiPageKind] = {
        "_rules": "rule",
        "rules": "rule",
        "decisions": "decision",
        "gotchas": "gotcha",
        "procedures": "procedure",
        "concepts": "concept",
    }
    return mapping.get(prefix, explicit_kind)


def _extract_wikilinks(content: str) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()
    for match in WIKILINK_RE.finditer(content):
        target = match.group(1).strip()
        if not target:
            continue
        try:
            norm = normalize_wiki_path(target)
        except ValueError:
            continue
        if norm not in seen:
            seen.add(norm)
            links.append(norm)
    return links


def _render_markdown(page: WikiPageRecord) -> str:
    """Render human-editable Markdown with YAML frontmatter (OKF / ai-memory compatible)."""
    frontmatter_lines = [
        "---",
        f"title: {json.dumps(page.title, ensure_ascii=False)}",
        f"kind: {page.kind}",
        f"tier: {page.tier}",
        f"project: {page.project}",
        f"pinned: {'true' if page.pinned else 'false'}",
        f"version: {page.version}",
        f"salience: {page.salience}",
        f"tags: {json.dumps(page.tags, ensure_ascii=False)}",
        f"entities: {json.dumps(page.entities, ensure_ascii=False)}",
        f"links: {json.dumps(page.links, ensure_ascii=False)}",
        f"updated_at: {page.updated_at}",
    ]
    if page.expires_at:
        frontmatter_lines.append(f"expires_at: {page.expires_at}")
    frontmatter_lines.append("---")
    return "\n".join(frontmatter_lines) + "\n\n" + page.content.rstrip() + "\n"


class WikiPageStore:
    """Markdown-first durable knowledge wiki with YAML frontmatter, wikilinks, versioning, and restore."""

    def __init__(self, store: FileStore, extractor: NonGenerativeExtractor | None = None):
        self.store = store
        self.extractor = extractor or NonGenerativeExtractor()
        self.bm25 = BM25Scorer()

    def _page_id(self, path: str) -> str:
        return digest(path.encode())[:16]

    def _meta_key(self, path: str) -> str:
        return f"wiki/meta/{self._page_id(path)}.json"

    def _history_key(self, path: str, version: int) -> str:
        return f"wiki/history/{self._page_id(path)}/v{version}.json"

    def _md_path(self, path: str):
        return self.store.path(f"wiki/pages/{path}")

    def _write_markdown_file(self, page: WikiPageRecord) -> None:
        md_file = self._md_path(page.path)
        md_file.parent.mkdir(parents=True, exist_ok=True)
        if page.deleted:
            if md_file.exists():
                md_file.unlink()
            return
        md_file.write_text(_render_markdown(page), encoding="utf-8")

    def write_page(self, req: WikiPageWriteRequest) -> WikiPageRecord:
        path = normalize_wiki_path(req.path)
        title, _ = scrub_secrets(req.title.strip())
        content, _ = scrub_secrets(req.content)
        if not title or not content.strip():
            raise ValueError("Wiki page title and content must not be empty")

        kind = _infer_kind_from_path(path, req.kind)
        entities = [e.normalized_name for e in self.extractor.extract_entities(f"{title}\n{content}")]
        links = _extract_wikilinks(content)
        now = utc_now_iso()

        with self.store.lock(f"wiki_{self._page_id(path)}"):
            meta_key = self._meta_key(path)
            if self.store.path(meta_key).exists():
                prev = WikiPageRecord.model_validate(self.store.get(meta_key))
                version = prev.version + 1
                created_at = prev.created_at
                access_count = prev.access_count
                last_accessed_at = prev.last_accessed_at
                salience = prev.salience
            else:
                version = 1
                created_at = now
                access_count = 0
                last_accessed_at = None
                salience = 1.0

            page = WikiPageRecord(
                path=path,
                title=title,
                content=content,
                kind=kind,
                tier=req.tier,
                project=req.project.strip() or "default",
                pinned=req.pinned,
                salience=salience,
                access_count=access_count,
                last_accessed_at=last_accessed_at,
                tags=list(dict.fromkeys(t.strip() for t in req.tags if t.strip())),
                entities=entities,
                links=links,
                version=version,
                deleted=False,
                author=req.author.strip() or "agent",
                created_at=created_at,
                updated_at=now,
                expires_at=req.expires_at,
            )
            self.store.put(self._history_key(path, version), page.model_dump())
            self.store.put(meta_key, page.model_dump())
            self._write_markdown_file(page)
            return page

    def patch_page(self, req: WikiPagePatchRequest) -> WikiPageRecord:
        path = normalize_wiki_path(req.path)
        with self.store.lock(f"wiki_{self._page_id(path)}"):
            meta_key = self._meta_key(path)
            if not self.store.path(meta_key).exists():
                raise FileNotFoundError(f"Wiki page not found: {path}")
            current = WikiPageRecord.model_validate(self.store.get(meta_key))
            if current.deleted:
                raise FileNotFoundError(f"Wiki page is deleted: {path}")

            new_content = current.content
            if req.replace_old is not None:
                if req.replace_old not in new_content:
                    raise ValueError("replace_old text not found in wiki page")
                new_content = new_content.replace(req.replace_old, req.replace_new or "", 1)
            if req.append:
                new_content = new_content.rstrip() + "\n\n" + req.append.strip()

            new_title = req.title.strip() if req.title else current.title
            new_pinned = req.pinned if req.pinned is not None else current.pinned
            new_tags = req.tags if req.tags is not None else current.tags

        return self.write_page(
            WikiPageWriteRequest(
                path=path,
                title=new_title,
                content=new_content,
                kind=current.kind,
                tier=current.tier,
                project=current.project,
                pinned=new_pinned,
                tags=new_tags,
                expires_at=current.expires_at,
                author=req.author,
            )
        )

    def read_page(self, raw_path: str, *, include_related: bool = True, bump_access: bool = True) -> dict[str, Any]:
        path = normalize_wiki_path(raw_path)
        meta_key = self._meta_key(path)
        if not self.store.path(meta_key).exists():
            raise FileNotFoundError(f"Wiki page not found: {path}")
        page = WikiPageRecord.model_validate(self.store.get(meta_key))
        if page.deleted:
            raise FileNotFoundError(f"Wiki page is deleted: {path}")

        if bump_access:
            with self.store.lock(f"wiki_{self._page_id(path)}"):
                page.access_count += 1
                page.last_accessed_at = utc_now_iso()
                self.store.put(meta_key, page.model_dump())

        related: list[dict[str, Any]] = []
        if include_related:
            all_pages = self.list_pages(project=page.project)
            by_path = {p.path: p for p in all_pages}
            linked_paths = set(page.links)
            # Also include backlinks pointing to this page
            for other in all_pages:
                if other.path != page.path and page.path in other.links:
                    linked_paths.add(other.path)
            for lp in sorted(linked_paths):
                neighbor = by_path.get(lp)
                if neighbor and not neighbor.deleted:
                    related.append(
                        {
                            "path": neighbor.path,
                            "title": neighbor.title,
                            "kind": neighbor.kind,
                            "version": neighbor.version,
                        }
                    )

        versions = self.list_versions(path)
        return {
            "page": page.model_dump(),
            "markdown": _render_markdown(page),
            "related": related,
            "versions": versions,
        }

    def list_versions(self, raw_path: str) -> list[int]:
        path = normalize_wiki_path(raw_path)
        hist_dir = self.store.path(f"wiki/history/{self._page_id(path)}")
        if not hist_dir.exists():
            return []
        nums: list[int] = []
        for f in hist_dir.glob("v*.json"):
            try:
                nums.append(int(f.stem.lstrip("v")))
            except ValueError:
                continue
        return sorted(nums)

    def list_pages(
        self,
        *,
        project: str | None = None,
        kind: str | None = None,
        include_deleted: bool = False,
        limit: int = 100,
    ) -> list[WikiPageRecord]:
        pages: list[WikiPageRecord] = []
        for path in self.store.root.glob("wiki/meta/*.json"):
            page = WikiPageRecord.model_validate(self.store.get(str(path.relative_to(self.store.root))))
            if page.deleted and not include_deleted:
                continue
            if project and page.project != project:
                continue
            if kind and page.kind != kind:
                continue
            pages.append(page)
        pages.sort(key=lambda p: (p.pinned, p.updated_at, p.path), reverse=True)
        return pages[:limit]

    def search_pages(self, query: str, *, project: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        pages = self.list_pages(project=project, limit=500)
        if not pages:
            return []
        by_path = {p.path: p for p in pages}
        docs = [
            (p.path, " ".join([p.title, p.path, p.content, *p.tags, *p.entities]))
            for p in pages
        ]
        scored = self.bm25.score(query, docs, limit=max(limit * 3, 20))
        weighted: list[tuple[WikiPageRecord, float]] = []
        for path, raw_score in scored:
            page = by_path[path]
            authority = KIND_AUTHORITY.get(page.kind, 1.0) * (1.15 if page.pinned else 1.0)
            weighted.append((page, round(raw_score * authority, 6)))
        weighted.sort(key=lambda item: (-item[1], item[0].path))
        return [
            {
                "path": page.path,
                "title": page.title,
                "kind": page.kind,
                "tier": page.tier,
                "project": page.project,
                "pinned": page.pinned,
                "version": page.version,
                "score": score,
                "preview": " ".join(page.content.split())[:220],
                "tags": page.tags,
            }
            for page, score in weighted[:limit]
        ]

    def delete_page(self, raw_path: str) -> dict[str, Any]:
        """Soft-delete a wiki page (retains version history for restore_page)."""
        path = normalize_wiki_path(raw_path)
        with self.store.lock(f"wiki_{self._page_id(path)}"):
            meta_key = self._meta_key(path)
            if not self.store.path(meta_key).exists():
                raise FileNotFoundError(f"Wiki page not found: {path}")
            page = WikiPageRecord.model_validate(self.store.get(meta_key))
            page.deleted = True
            page.updated_at = utc_now_iso()
            self.store.put(meta_key, page.model_dump())
            self._write_markdown_file(page)
            return {"deleted": True, "path": path, "recoverable_versions": self.list_versions(path)}

    def restore_page(self, raw_path: str, *, version: int | None = None) -> WikiPageRecord:
        """Restore a soft-deleted page or roll back to a previous retained version."""
        path = normalize_wiki_path(raw_path)
        with self.store.lock(f"wiki_{self._page_id(path)}"):
            meta_key = self._meta_key(path)
            if not self.store.path(meta_key).exists():
                raise FileNotFoundError(f"Wiki page not found: {path}")
            current = WikiPageRecord.model_validate(self.store.get(meta_key))
            versions = self.list_versions(path)
            if not versions:
                raise FileNotFoundError(f"No retained history for wiki page: {path}")
            target_ver = version if version is not None else versions[-1]
            hist_key = self._history_key(path, target_ver)
            if not self.store.path(hist_key).exists():
                raise FileNotFoundError(f"Wiki page version v{target_ver} not found for {path}")
            snapshot = WikiPageRecord.model_validate(self.store.get(hist_key))
            next_version = max(current.version, versions[-1]) + 1
            now = utc_now_iso()
            restored = snapshot.model_copy(
                update={
                    "version": next_version,
                    "deleted": False,
                    "updated_at": now,
                }
            )
            self.store.put(self._history_key(path, next_version), restored.model_dump())
            self.store.put(meta_key, restored.model_dump())
            self._write_markdown_file(restored)
            return restored
