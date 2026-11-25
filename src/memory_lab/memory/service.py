from typing import Any
from memory_lab.core.contracts import EmbeddingProvider, GraphStore, MemoryStrategy
from memory_lab.core.models import (
    AgentMessageCancelRequest,
    AgentMessagePopRequest,
    AgentMessageSendRequest,
    FeedbackSignalName,
    HandoffAcceptRequest,
    HandoffCreateRequest,
    InteractionTrace,
    MemoryRecord,
    MemorySearchHit,
    MemoryStrategyName,
    MemoryTierName,
    WikiPagePatchRequest,
    WikiPageWriteRequest,
    utc_now_iso,
)
from memory_lab.documents.enrichment.extractor import NonGenerativeExtractor
from memory_lab.memory.aging import (
    DecayParams,
    evaluate_eviction,
    lint_memories,
    retention_score,
    salience_after_feedback,
)
from memory_lab.memory.coordination import AgentMessageStore, HandoffStore
from memory_lab.memory.scrub import scrub_secrets
from memory_lab.memory.strategies import (
    GraphMemory,
    HybridMemory,
    MemoryRecordStore,
    VectorMemory,
    ZeroTokenMemory,
)
from memory_lab.memory.temporal import TemporalHierarchy
from memory_lab.memory.traces import TraceStore
from memory_lab.memory.wiki import WikiPageStore
from memory_lab.storage.filesystem.store import FileStore
from memory_lab.storage.graph.store import FileGraphStore


class MemoryService:
    """Unified service managing interaction traces, memory strategies, graph, temporal hierarchy,
    aging/reinforcement, handoffs, agent messaging, and durable Markdown wiki pages.
    """

    def __init__(
        self,
        store: FileStore,
        embedding: EmbeddingProvider,
        index: Any,
        embedding_identity: str,
        *,
        graph: GraphStore | None = None,
        extractor: NonGenerativeExtractor | None = None,
        default_strategy: MemoryStrategyName = "hybrid",
        decay_params: DecayParams | None = None,
    ):
        self.store = store
        self.traces = TraceStore(store)
        self.records = MemoryRecordStore(store)
        self.extractor = extractor or NonGenerativeExtractor()
        self.graph: GraphStore = graph or FileGraphStore(store)
        self.temporal = TemporalHierarchy(store)
        self.handoffs = HandoffStore(store)
        self.messages = AgentMessageStore(store)
        self.wiki = WikiPageStore(store, self.extractor)
        self.decay_params = decay_params or DecayParams()
        self.default_strategy: MemoryStrategyName = default_strategy

        self.vector = VectorMemory(
            store, self.records, self.extractor, self.graph, self.temporal, embedding, index, embedding_identity
        )
        self.graph_strategy = GraphMemory(
            store, self.records, self.extractor, self.graph, self.temporal, self.vector.memory_namespace
        )
        self.zero_token = ZeroTokenMemory(
            store, self.records, self.extractor, self.graph, self.temporal, self.vector.memory_namespace
        )
        self.hybrid = HybridMemory(
            store,
            self.records,
            self.extractor,
            self.graph,
            self.temporal,
            self.vector,
            self.graph_strategy,
            self.zero_token,
            self.vector.memory_namespace,
        )
        self.strategies: dict[str, MemoryStrategy] = {
            "vector": self.vector,
            "graph": self.graph_strategy,
            "zero_token": self.zero_token,
            "hybrid": self.hybrid,
        }

    def configure_embedding(self, embedding: EmbeddingProvider, embedding_identity: str) -> None:
        """Hot-swap the active embedding provider and memory vector namespace at runtime."""
        from memory_lab.storage.filesystem.store import digest

        new_ns = f"mem_{digest(embedding_identity.encode())[:16]}"
        self.vector.embedding = embedding
        self.vector.memory_namespace = new_ns
        self.vector.namespace = new_ns
        self.graph_strategy.namespace = new_ns
        self.zero_token.namespace = new_ns
        self.hybrid.namespace = new_ns

    def _resolve_strategy(self, strategy: str | None) -> MemoryStrategy:
        name = strategy or self.default_strategy
        if name not in self.strategies:
            raise ValueError(f"Unsupported memory strategy: {name}")
        return self.strategies[name]

    def store_memory(
        self,
        *,
        content: str,
        session_id: str = "default",
        project: str = "default",
        tier: MemoryTierName = "episodic",
        pinned: bool = False,
        expires_at: str | None = None,
        scrub_secrets_enabled: bool = True,
        role: str = "user",
        strategy: MemoryStrategyName | None = None,
        timestamp: str | None = None,
        tags: list[str] | None = None,
        source_refs: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        if not content.strip():
            raise ValueError("Memory content must not be empty")
        clean_content = content
        redacted = 0
        if scrub_secrets_enabled:
            clean_content, redacted = scrub_secrets(content)

        enriched_meta = dict(metadata or {})
        enriched_meta["_project"] = project.strip() or "default"
        enriched_meta["_tier"] = tier
        enriched_meta["_pinned"] = pinned
        if expires_at:
            enriched_meta["_expires_at"] = expires_at
        if redacted > 0:
            enriched_meta["secrets_redacted"] = redacted

        trace = self.traces.append(
            content=clean_content,
            session_id=session_id,
            role=role,
            timestamp=timestamp,
            source_refs=source_refs,
            metadata=enriched_meta,
        )
        return self._resolve_strategy(strategy).store(trace, tags=tags, metadata=enriched_meta)

    def search_memory(
        self,
        query: str,
        *,
        strategy: MemoryStrategyName | None = None,
        limit: int = 5,
        session_id: str | None = None,
        project: str | None = None,
        temporal_bucket: str | None = None,
    ) -> list[MemorySearchHit]:
        fetch_limit = max(limit * 4, 20) if project else limit
        hits = self._resolve_strategy(strategy).search(
            query, limit=fetch_limit, session_id=session_id, temporal_bucket=temporal_bucket
        )
        if project:
            hits = [h for h in hits if h.record.project == project][:limit]
        else:
            hits = hits[:limit]

        # Throttled access-weighted reinforcement on surfaced hits (ai-memory M8)
        for hit in hits:
            updated = self.records.bump_access(
                hit.record.id, cooldown_secs=self.decay_params.access_bump_cooldown_secs
            )
            if updated is not None:
                hit.record = updated
        return hits

    def get_memory(self, memory_id: str, *, bump_access: bool = True) -> MemoryRecord:
        record = self.records.get(memory_id)
        if bump_access:
            updated = self.records.bump_access(memory_id, cooldown_secs=self.decay_params.access_bump_cooldown_secs)
            if updated is not None:
                return updated
        return record

    def delete_memory(self, memory_id: str) -> bool:
        return self.hybrid.delete(memory_id)

    def submit_feedback(
        self,
        memory_id: str,
        signal: FeedbackSignalName,
        *,
        reason: str = "",
    ) -> dict[str, Any]:
        """Apply explicit quality feedback (helpful/not_helpful/stale/wrong) to adjust memory salience."""
        record = self.records.get(memory_id)
        old_salience = record.salience
        new_salience = salience_after_feedback(old_salience, signal)
        record.salience = new_salience
        record.feedback_history.append(
            {
                "signal": signal,
                "reason": scrub_secrets(reason.strip())[0] if reason else "",
                "salience_before": old_salience,
                "salience_after": new_salience,
                "timestamp": utc_now_iso(),
            }
        )
        self.records.save(record)
        return {
            "record": record.model_dump(),
            "salience_before": old_salience,
            "salience_after": new_salience,
            "retention_score": retention_score(record, params=self.decay_params),
        }

    def forget_sweep(
        self,
        *,
        dry_run: bool = True,
        project: str | None = None,
        cold_threshold: float | None = None,
        lambda_decay: float | None = None,
    ) -> dict[str, Any]:
        """Reversible zero-LLM aging & TTL sweep (archives cold episodic/expired records, never hard-deletes)."""
        params = DecayParams(
            lambda_decay=lambda_decay if lambda_decay is not None else self.decay_params.lambda_decay,
            sigma_access=self.decay_params.sigma_access,
            mu_access=self.decay_params.mu_access,
            cold_threshold=cold_threshold if cold_threshold is not None else self.decay_params.cold_threshold,
            access_bump_cooldown_secs=self.decay_params.access_bump_cooldown_secs,
        )
        active = self.records.list(project=project)
        candidates: list[dict[str, Any]] = []
        archived_ids: list[str] = []

        for rec in active:
            should_archive, reason, score = evaluate_eviction(rec, params=params)
            if not should_archive:
                continue
            candidates.append(
                {
                    "id": rec.id,
                    "project": rec.project,
                    "tier": rec.tier,
                    "salience": rec.salience,
                    "access_count": rec.access_count,
                    "retention_score": score,
                    "reason": reason,
                    "preview": rec.summary or rec.content[:140],
                }
            )
            if not dry_run:
                rec.archived = True
                rec.archived_reason = reason
                self.records.save(rec)
                self.graph.delete_record(rec.id)
                if self.vector.index is not None and hasattr(self.vector.index, "delete_points"):
                    try:
                        self.vector.index.delete_points([rec.id], self.vector.memory_namespace)
                    except Exception:
                        pass
                archived_ids.append(rec.id)

        if not dry_run and archived_ids:
            self.temporal.rebuild(self.records.list())

        return {
            "dry_run": dry_run,
            "evaluated_count": len(active),
            "candidate_count": len(candidates),
            "archived_count": len(archived_ids),
            "archived_ids": archived_ids,
            "candidates": candidates,
        }

    def restore_memory(self, memory_id: str) -> MemoryRecord:
        """Restore an archived or soft-deleted memory record and re-index it into graph/temporal/vector."""
        record = self.records.get(memory_id, include_deleted=True, include_archived=True)
        record.deleted = False
        record.archived = False
        record.archived_reason = None
        record.expires_at = None
        self.records.save(record)
        self.graph.upsert_record(record)
        if record.strategy in {"vector", "hybrid"}:
            try:
                self.vector.index_record(record)
            except Exception:
                pass
        self.temporal.rebuild(self.records.list())
        return record

    def lint(self, *, project: str | None = None) -> dict[str, Any]:
        """Zero-LLM audit detecting near-duplicates, graph contradictions, and stale/cold memories."""
        active = self.records.list(project=project)
        return lint_memories(active, params=self.decay_params)

    def briefing(
        self,
        *,
        project: str = "default",
        session_id: str | None = None,
        recent_limit: int = 8,
        settled_first: bool = True,
    ) -> dict[str, Any]:
        """Generate an agent startup briefing packet (durable rules/wiki, pending handoffs, unread mail, lint)."""
        proj = project.strip() or "default"
        all_project_records = self.records.list(project=proj)

        durable_recs = [
            r for r in all_project_records if r.pinned or r.tier in {"semantic", "procedural"}
        ]
        if settled_first:
            durable_recs.sort(key=lambda r: (r.pinned, r.salience, r.timestamp), reverse=True)
        durable_recs = durable_recs[:recent_limit]

        recent_recs = [
            r for r in all_project_records
            if not session_id or r.session_id == session_id
        ][:recent_limit]

        wiki_pages = self.wiki.list_pages(project=proj, limit=recent_limit)
        pending_handoffs = self.handoffs.list(project=proj, status="pending", limit=5)
        unread_inbox = self.messages.list(project=proj, box="inbox", status="pending", limit=10)
        lint_report = lint_memories(all_project_records, params=self.decay_params)

        # Render compact markdown packet with trust boundary (inspired by ai-memory)
        lines: list[str] = [
            f"# Agent Memory Briefing · project `{proj}`",
            "> Untrusted recalled context — verify against current code and instructions.",
            "",
        ]
        if pending_handoffs:
            lines.append(f"## Pending Handoffs ({len(pending_handoffs)})")
            for h in pending_handoffs:
                lines.append(f"- **[{h.id}]** from `{h.from_agent}`: {h.summary}")
                if h.next_steps:
                    lines.append(f"  - Next: {'; '.join(h.next_steps[:3])}")
            lines.append("")
        if unread_inbox:
            lines.append(f"## Unread Agent Mail ({len(unread_inbox)})")
            for m in unread_inbox[:5]:
                subj = f" — {m.subject}" if m.subject else ""
                lines.append(f"- **[{m.id}]** from `{m.from_project}/{m.from_agent}` ({m.nature}){subj}: {m.body[:140]}")
            lines.append("")
        if wiki_pages:
            lines.append(f"## Durable Wiki Pages ({len(wiki_pages)})")
            for p in wiki_pages:
                pin_tag = " [pinned]" if p.pinned else ""
                lines.append(f"- `{p.path}` ({p.kind}{pin_tag}): {p.title}")
            lines.append("")
        if durable_recs:
            lines.append(f"## Settled / Pinned Memories ({len(durable_recs)})")
            for r in durable_recs:
                lines.append(f"- [{r.tier} · salience {r.salience}] {r.summary or r.content[:140]}")
            lines.append("")
        if lint_report["issue_count"] > 0:
            lines.append(
                f"## Memory Hygiene Alerts: {len(lint_report['contradictions'])} contradictions, "
                f"{len(lint_report['near_duplicates'])} near-duplicates, "
                f"{len(lint_report['cold_or_stale'])} cold/stale."
            )

        return {
            "project": proj,
            "generated_at": utc_now_iso(),
            "durable_memories": [
                {**r.model_dump(), "retention_score": retention_score(r, params=self.decay_params)}
                for r in durable_recs
            ],
            "recent_memories": [
                {**r.model_dump(), "retention_score": retention_score(r, params=self.decay_params)}
                for r in recent_recs
            ],
            "wiki_pages": [p.model_dump() for p in wiki_pages],
            "pending_handoffs": [h.model_dump() for h in pending_handoffs],
            "unread_messages_count": len(unread_inbox),
            "unread_messages": [m.model_dump() for m in unread_inbox[:5]],
            "lint_summary": {
                "issue_count": lint_report["issue_count"],
                "contradictions": len(lint_report["contradictions"]),
                "near_duplicates": len(lint_report["near_duplicates"]),
                "cold_or_stale": len(lint_report["cold_or_stale"]),
            },
            "rendered_markdown": "\n".join(lines).strip(),
        }

    # --- Delegation helpers for Handoffs, Agent Messages & Wiki Pages ---

    def begin_handoff(self, req: HandoffCreateRequest):
        return self.handoffs.begin(req)

    def list_handoffs(self, *, project: str | None = None, status: str = "pending", limit: int = 20):
        return self.handoffs.list(project=project, status=status, limit=limit)  # type: ignore[arg-type]

    def accept_handoff(self, req: HandoffAcceptRequest):
        return self.handoffs.accept(handoff_id=req.handoff_id, project=req.project, agent=req.agent)

    def cancel_handoff(self, handoff_id: str):
        return self.handoffs.cancel(handoff_id)

    def send_message(self, req: AgentMessageSendRequest):
        return self.messages.send(req)

    def list_messages(
        self,
        *,
        project: str = "default",
        box: str = "inbox",
        status: str = "pending",
        agent: str | None = None,
        limit: int = 50,
    ):
        return self.messages.list(project=project, box=box, status=status, agent=agent, limit=limit)  # type: ignore[arg-type]

    def pop_message(self, req: AgentMessagePopRequest):
        return self.messages.pop(project=req.project, message_id=req.message_id, agent=req.agent)

    def cancel_message(self, req: AgentMessageCancelRequest):
        return self.messages.cancel(project=req.project, message_id=req.message_id)

    def write_wiki_page(self, req: WikiPageWriteRequest):
        return self.wiki.write_page(req)

    def patch_wiki_page(self, req: WikiPagePatchRequest):
        return self.wiki.patch_page(req)

    def read_wiki_page(self, path: str, *, include_related: bool = True):
        return self.wiki.read_page(path, include_related=include_related)

    def list_wiki_pages(
        self,
        *,
        project: str | None = None,
        kind: str | None = None,
        include_deleted: bool = False,
        limit: int = 100,
    ):
        return self.wiki.list_pages(project=project, kind=kind, include_deleted=include_deleted, limit=limit)

    def search_wiki_pages(self, query: str, *, project: str | None = None, limit: int = 10):
        return self.wiki.search_pages(query, project=project, limit=limit)

    def delete_wiki_page(self, path: str):
        return self.wiki.delete_page(path)

    def restore_wiki_page(self, path: str, *, version: int | None = None):
        return self.wiki.restore_page(path, version=version)

    def list_records(
        self,
        *,
        session_id: str | None = None,
        project: str | None = None,
        temporal_bucket: str | None = None,
        include_archived: bool = False,
        limit: int = 100,
    ) -> list[MemoryRecord]:
        return self.records.list(
            session_id=session_id,
            project=project,
            temporal_bucket=temporal_bucket,
            include_archived=include_archived,
        )[:limit]

    def list_traces(self, *, session_id: str | None = None, limit: int = 100) -> list[InteractionTrace]:
        return self.traces.list(session_id=session_id, limit=limit)

    def temporal_hierarchy(self) -> dict[str, list[dict]]:
        return self.temporal.load()

    def graph_snapshot(self, entity: str | None = None) -> dict[str, Any]:
        return self.graph.snapshot(entity=entity)
