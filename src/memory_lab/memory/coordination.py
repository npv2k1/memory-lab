from typing import Literal
from uuid import uuid4
from memory_lab.core.models import (
    AgentMessageRecord,
    AgentMessageSendRequest,
    HandoffCreateRequest,
    HandoffRecord,
    utc_now_iso,
)
from memory_lab.memory.scrub import scrub_secrets
from memory_lab.storage.filesystem.store import FileStore

MAX_PENDING_INBOX = 256


class HandoffStore:
    """Cross-agent claim-once session handoff store inspired by akitaonrails/ai-memory."""

    def __init__(self, store: FileStore):
        self.store = store

    def _key(self, handoff_id: str) -> str:
        return f"memory/handoffs/{handoff_id}.json"

    def begin(self, req: HandoffCreateRequest) -> HandoffRecord:
        summary, _ = scrub_secrets(req.summary.strip())
        if not summary:
            raise ValueError("Handoff summary must not be empty")
        next_steps = [scrub_secrets(s.strip())[0] for s in req.next_steps if s.strip()]
        open_questions = [scrub_secrets(q.strip())[0] for q in req.open_questions if q.strip()]
        files_touched = [f.strip() for f in req.files_touched if f.strip()]

        record = HandoffRecord(
            id=uuid4().hex[:12],
            project=req.project.strip() or "default",
            session_id=req.session_id.strip() or "default",
            from_agent=req.from_agent.strip() or "agent",
            summary=summary,
            next_steps=next_steps,
            open_questions=open_questions,
            files_touched=files_touched,
            shared=req.shared,
            status="pending",
        )
        with self.store.lock("memory_handoffs"):
            self.store.put(self._key(record.id), record.model_dump())
        return record

    def _all_unlocked(self) -> list[HandoffRecord]:
        items: list[HandoffRecord] = []
        for path in self.store.root.glob("memory/handoffs/*.json"):
            items.append(HandoffRecord.model_validate(self.store.get(str(path.relative_to(self.store.root)))))
        items.sort(key=lambda item: (item.created_at, item.id))
        return items

    def list(
        self,
        *,
        project: str | None = None,
        status: Literal["pending", "accepted", "cancelled", "all"] = "pending",
        limit: int = 20,
    ) -> list[HandoffRecord]:
        items = self._all_unlocked()
        filtered = [
            item
            for item in items
            if (not project or item.project == project)
            and (status == "all" or item.status == status)
        ]
        # Newest first for listing
        filtered.reverse()
        return filtered[:limit]

    def accept(
        self,
        *,
        handoff_id: str | None = None,
        project: str | None = None,
        agent: str = "agent",
    ) -> HandoffRecord:
        """Atomically claim a pending handoff once. Raises ValueError if already claimed or none pending."""
        with self.store.lock("memory_handoffs"):
            if handoff_id:
                key = self._key(handoff_id.strip())
                if not self.store.path(key).exists():
                    raise FileNotFoundError(f"Handoff not found: {handoff_id}")
                target = HandoffRecord.model_validate(self.store.get(key))
                if target.status != "pending":
                    raise ValueError(f"Handoff {target.id} is already {target.status}")
            else:
                pending = [
                    item
                    for item in self._all_unlocked()
                    if item.status == "pending" and (not project or item.project == project)
                ]
                if not pending:
                    raise FileNotFoundError("No pending handoff available to accept")
                # Claim newest pending handoff for the project
                target = pending[-1]

            target.status = "accepted"
            target.accepted_at = utc_now_iso()
            target.accepted_by = agent.strip() or "agent"
            self.store.put(self._key(target.id), target.model_dump())
            return target

    def cancel(self, handoff_id: str) -> HandoffRecord:
        with self.store.lock("memory_handoffs"):
            key = self._key(handoff_id.strip())
            if not self.store.path(key).exists():
                raise FileNotFoundError(f"Handoff not found: {handoff_id}")
            target = HandoffRecord.model_validate(self.store.get(key))
            if target.status != "pending":
                raise ValueError(f"Cannot cancel handoff {target.id}: status is {target.status}")
            target.status = "cancelled"
            target.cancelled_at = utc_now_iso()
            self.store.put(key, target.model_dump())
            return target


class AgentMessageStore:
    """Directed, claim-once cross-project agent inbox/outbox with epistemic provenance."""

    def __init__(self, store: FileStore):
        self.store = store

    def _key(self, msg_id: str) -> str:
        return f"memory/messages/{msg_id}.json"

    def _all_unlocked(self) -> list[AgentMessageRecord]:
        items: list[AgentMessageRecord] = []
        for path in self.store.root.glob("memory/messages/*.json"):
            items.append(AgentMessageRecord.model_validate(self.store.get(str(path.relative_to(self.store.root)))))
        items.sort(key=lambda item: (item.created_at, item.id))
        return items

    def send(self, req: AgentMessageSendRequest) -> AgentMessageRecord:
        body, _ = scrub_secrets(req.body.strip())
        if not body:
            raise ValueError("Message body must not be empty")
        evidence, _ = scrub_secrets(req.evidence.strip())
        if req.nature == "measured" and not evidence:
            raise ValueError("Measured claims require non-empty supporting evidence")
        subject, _ = scrub_secrets(req.subject.strip())

        to_project = req.to_project.strip()
        from_project = req.from_project.strip() or "default"
        to_agent = req.to_agent.strip() if req.to_agent and req.to_agent.strip() else None

        with self.store.lock("memory_messages"):
            pending_count = sum(
                1 for m in self._all_unlocked() if m.to_project == to_project and m.status == "pending"
            )
            if pending_count >= MAX_PENDING_INBOX:
                raise ValueError(f"Recipient inbox for project '{to_project}' is full ({MAX_PENDING_INBOX} pending)")

            msg = AgentMessageRecord(
                id=uuid4().hex[:12],
                from_project=from_project,
                to_project=to_project,
                from_agent=req.from_agent.strip() or "agent",
                to_agent=to_agent,
                subject=subject,
                body=body,
                nature=req.nature,
                evidence=evidence,
                status="pending",
            )
            self.store.put(self._key(msg.id), msg.model_dump())
            return msg

    def list(
        self,
        *,
        project: str,
        box: Literal["inbox", "outbox"] = "inbox",
        status: Literal["pending", "claimed", "cancelled", "all"] = "pending",
        agent: str | None = None,
        limit: int = 50,
    ) -> list[AgentMessageRecord]:
        """Read-only mailbox inspection — never claims messages."""
        items = self._all_unlocked()
        result: list[AgentMessageRecord] = []
        for msg in items:
            if box == "inbox":
                if msg.to_project != project:
                    continue
                if agent and msg.to_agent and msg.to_agent != agent:
                    continue
            else:
                if msg.from_project != project:
                    continue
            if status != "all" and msg.status != status:
                continue
            result.append(msg)
        return result[:limit]

    def pop(
        self,
        *,
        project: str,
        message_id: str | None = None,
        agent: str | None = None,
    ) -> AgentMessageRecord | None:
        """Atomically claim one pending inbox message (oldest FIFO, or specific message_id)."""
        if message_id is not None and not message_id.strip():
            raise ValueError("message_id must not be empty whitespace")

        with self.store.lock("memory_messages"):
            if message_id:
                key = self._key(message_id.strip())
                if not self.store.path(key).exists():
                    raise FileNotFoundError(f"Message not found: {message_id}")
                target = AgentMessageRecord.model_validate(self.store.get(key))
                if target.to_project != project:
                    raise FileNotFoundError(f"Message {target.id} is not in project '{project}' inbox")
                if target.status != "pending":
                    return None
                if target.to_agent and target.to_agent != agent:
                    raise ValueError(f"Message {target.id} is addressed specifically to agent '{target.to_agent}'")
            else:
                candidates = [
                    m
                    for m in self._all_unlocked()
                    if m.to_project == project
                    and m.status == "pending"
                    and (m.to_agent is None or m.to_agent == agent)
                ]
                if not candidates:
                    return None
                target = candidates[0]  # FIFO oldest pending

            target.status = "claimed"
            target.claimed_at = utc_now_iso()
            target.claimed_by = (agent or "agent").strip()
            self.store.put(self._key(target.id), target.model_dump())
            return target

    def cancel(self, *, project: str, message_id: str | None = None) -> dict:
        """Cancel a specific pending outbox message, or all pending messages sent from `project`."""
        if message_id is not None and not message_id.strip():
            raise ValueError("message_id must not be empty whitespace")

        cancelled_ids: list[str] = []
        now = utc_now_iso()
        with self.store.lock("memory_messages"):
            if message_id:
                key = self._key(message_id.strip())
                if not self.store.path(key).exists():
                    raise FileNotFoundError(f"Message not found: {message_id}")
                target = AgentMessageRecord.model_validate(self.store.get(key))
                if target.from_project != project:
                    raise FileNotFoundError(f"Message {target.id} was not sent from project '{project}'")
                if target.status != "pending":
                    raise ValueError(f"Cannot cancel message {target.id}: already {target.status}")
                target.status = "cancelled"
                target.cancelled_at = now
                self.store.put(key, target.model_dump())
                cancelled_ids.append(target.id)
            else:
                for msg in self._all_unlocked():
                    if msg.from_project == project and msg.status == "pending":
                        msg.status = "cancelled"
                        msg.cancelled_at = now
                        self.store.put(self._key(msg.id), msg.model_dump())
                        cancelled_ids.append(msg.id)
        return {"cancelled_ids": cancelled_ids, "count": len(cancelled_ids)}
