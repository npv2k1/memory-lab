import json
from typing import Any
from memory_lab.core.models import (
    AgentMessageCancelRequest,
    AgentMessagePopRequest,
    AgentMessageSendRequest,
    HandoffAcceptRequest,
    HandoffCreateRequest,
    WikiPagePatchRequest,
    WikiPageWriteRequest,
)
from memory_lab.memory.service import MemoryService

SUPPORTED_PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")
DEFAULT_PROTOCOL_VERSION = "2025-03-26"

MUTATING_MCP_TOOLS: frozenset[str] = frozenset({
    "memory_store",
    "memory_delete",
    "memory_feedback",
    "memory_forget_sweep",
    "memory_restore",
    "memory_handoff_begin",
    "memory_handoff_accept",
    "memory_handoff_cancel",
    "memory_message_send",
    "memory_message_pop",
    "memory_message_cancel",
    "memory_write_page",
    "memory_patch_page",
    "memory_delete_page",
    "memory_restore_page",
})

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "memory_store",
        "description": "Store an agent interaction trace and index it using the selected memory strategy (auto-scrubs secrets).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Interaction or memory text content."},
                "session_id": {"type": "string", "default": "default"},
                "project": {"type": "string", "default": "default"},
                "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"], "default": "episodic"},
                "pinned": {"type": "boolean", "default": False},
                "expires_at": {"type": "string", "description": "Optional ISO-8601 UTC expiration timestamp."},
                "scrub_secrets": {"type": "boolean", "default": True},
                "role": {"type": "string", "enum": ["user", "assistant", "system", "tool"], "default": "user"},
                "strategy": {"type": "string", "enum": ["vector", "graph", "zero_token", "hybrid"]},
                "timestamp": {"type": "string", "description": "Optional ISO-8601 UTC timestamp."},
                "tags": {"type": "array", "items": {"type": "string"}},
                "source_refs": {"type": "array", "items": {"type": "string"}},
                "metadata": {"type": "object"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "memory_search",
        "description": "Search stored memories using vector, graph, zero_token, or hybrid retrieval (bumps access reinforcement).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "strategy": {"type": "string", "enum": ["vector", "graph", "zero_token", "hybrid"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
                "session_id": {"type": "string"},
                "project": {"type": "string"},
                "temporal_bucket": {"type": "string", "description": "Optional YYYY, YYYY-MM, or YYYY-MM-DD filter."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_get",
        "description": "Retrieve a stored memory record by ID (reinforces access counter).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Unique memory record ID."},
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "memory_delete",
        "description": "Delete a memory record by ID and remove it from active indexes (recoverable via memory_restore).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Unique memory record ID."},
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "memory_briefing",
        "description": "Generate a startup agent briefing packet (durable memories, wiki pages, pending handoffs, unread mail, lint alerts).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {"type": "string", "default": "default"},
                "session_id": {"type": "string"},
                "recent_limit": {"type": "integer", "minimum": 1, "maximum": 30, "default": 8},
                "settled_first": {"type": "boolean", "default": True},
            },
        },
    },
    {
        "name": "memory_feedback",
        "description": "Provide explicit quality feedback (helpful/not_helpful/stale/wrong) to adjust a memory's salience.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "signal": {"type": "string", "enum": ["helpful", "not_helpful", "stale", "wrong"]},
                "reason": {"type": "string", "default": ""},
            },
            "required": ["memory_id", "signal"],
        },
    },
    {
        "name": "memory_forget_sweep",
        "description": "Run zero-LLM aging and TTL eviction sweep (reversible archive; dry_run=true by default).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dry_run": {"type": "boolean", "default": True},
                "project": {"type": "string"},
                "cold_threshold": {"type": "number", "default": 0.20},
                "lambda_decay": {"type": "number", "default": 0.02},
            },
        },
    },
    {
        "name": "memory_restore",
        "description": "Restore an archived or soft-deleted memory record and re-index it.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "memory_lint",
        "description": "Zero-LLM memory hygiene audit: detects near-duplicates, graph contradictions, and stale/cold records.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {"type": "string"},
            },
        },
    },
    {
        "name": "memory_handoff_begin",
        "description": "Create a claim-once cross-agent session handoff packet.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "project": {"type": "string", "default": "default"},
                "session_id": {"type": "string", "default": "default"},
                "from_agent": {"type": "string", "default": "agent"},
                "next_steps": {"type": "array", "items": {"type": "string"}},
                "open_questions": {"type": "array", "items": {"type": "string"}},
                "files_touched": {"type": "array", "items": {"type": "string"}},
                "shared": {"type": "boolean", "default": True},
            },
            "required": ["summary"],
        },
    },
    {
        "name": "memory_handoff_list",
        "description": "List cross-agent handoffs (read-only).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "accepted", "cancelled", "all"], "default": "pending"},
                "limit": {"type": "integer", "default": 20},
            },
        },
    },
    {
        "name": "memory_handoff_accept",
        "description": "Atomically claim a pending handoff once (pending -> accepted).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "handoff_id": {"type": "string"},
                "project": {"type": "string"},
                "agent": {"type": "string", "default": "agent"},
            },
        },
    },
    {
        "name": "memory_handoff_cancel",
        "description": "Cancel a pending handoff.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "handoff_id": {"type": "string"},
            },
            "required": ["handoff_id"],
        },
    },
    {
        "name": "memory_message_send",
        "description": "Send a directed claim-once message to another project/agent inbox (supports epistemic nature + evidence).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "to_project": {"type": "string"},
                "body": {"type": "string"},
                "from_project": {"type": "string", "default": "default"},
                "from_agent": {"type": "string", "default": "agent"},
                "to_agent": {"type": "string"},
                "subject": {"type": "string", "default": ""},
                "nature": {"type": "string", "enum": ["measured", "reported", "inferred", "unknown"], "default": "unknown"},
                "evidence": {"type": "string", "default": ""},
            },
            "required": ["to_project", "body"],
        },
    },
    {
        "name": "memory_message_list",
        "description": "Read-only listing of an agent project's inbox or outbox (never claims messages).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {"type": "string", "default": "default"},
                "box": {"type": "string", "enum": ["inbox", "outbox"], "default": "inbox"},
                "status": {"type": "string", "enum": ["pending", "claimed", "cancelled", "all"], "default": "pending"},
                "agent": {"type": "string"},
                "limit": {"type": "integer", "default": 50},
            },
        },
    },
    {
        "name": "memory_message_pop",
        "description": "Atomically claim one pending message from a project inbox (claim-once FIFO).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {"type": "string", "default": "default"},
                "message_id": {"type": "string"},
                "agent": {"type": "string"},
            },
        },
    },
    {
        "name": "memory_message_cancel",
        "description": "Cancel a pending sent message (or all pending outbox messages from a project).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {"type": "string", "default": "default"},
                "message_id": {"type": "string"},
            },
        },
    },
    {
        "name": "memory_write_page",
        "description": "Write or update a versioned Markdown durable wiki page (concepts/, decisions/, gotchas/, procedures/, _rules/).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "title": {"type": "string"},
                "content": {"type": "string"},
                "kind": {"type": "string", "enum": ["concept", "decision", "gotcha", "procedure", "rule", "note"], "default": "note"},
                "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"], "default": "semantic"},
                "project": {"type": "string", "default": "default"},
                "pinned": {"type": "boolean", "default": False},
                "tags": {"type": "array", "items": {"type": "string"}},
                "expires_at": {"type": "string"},
                "author": {"type": "string", "default": "agent"},
            },
            "required": ["path", "title", "content"],
        },
    },
    {
        "name": "memory_patch_page",
        "description": "Append or replace content in an existing durable wiki page (creates a new retained version).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "append": {"type": "string"},
                "replace_old": {"type": "string"},
                "replace_new": {"type": "string"},
                "title": {"type": "string"},
                "pinned": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "author": {"type": "string", "default": "agent"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "memory_read_page",
        "description": "Read a durable wiki page along with its rendered Markdown, wikilink neighbours, and version history.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "include_related": {"type": "boolean", "default": True},
            },
            "required": ["path"],
        },
    },
    {
        "name": "memory_list_pages",
        "description": "List or search durable Markdown wiki pages.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {"type": "string"},
                "kind": {"type": "string"},
                "query": {"type": "string"},
                "include_deleted": {"type": "boolean", "default": False},
                "limit": {"type": "integer", "default": 50},
            },
        },
    },
    {
        "name": "memory_delete_page",
        "description": "Soft-delete a wiki page (history retained; recoverable via memory_restore_page).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "memory_restore_page",
        "description": "Restore a soft-deleted wiki page or roll back to a specific retained version.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "version": {"type": "integer", "minimum": 1},
            },
            "required": ["path"],
        },
    },
]


class MCPServer:
    """JSON-RPC 2.0 Model Context Protocol adapter for MemoryService."""

    def __init__(self, memory: MemoryService, *, server_name: str = "memory-lab", server_version: str = "0.3.0"):
        self.memory = memory
        self.server_name = server_name
        self.server_version = server_version

    def _negotiate_protocol(self, requested: str | None) -> str:
        if requested and requested in SUPPORTED_PROTOCOL_VERSIONS:
            return requested
        return DEFAULT_PROTOCOL_VERSION

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        req_id = message.get("id")
        method = message.get("method")
        params = message.get("params") or {}

        if not method:
            return self._error(req_id, -32600, "Invalid Request: missing method")

        if method == "notifications/initialized":
            return None
        if method == "ping":
            return self._result(req_id, {})
        if method == "initialize":
            negotiated = self._negotiate_protocol(params.get("protocolVersion"))
            return self._result(
                req_id,
                {
                    "protocolVersion": negotiated,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": self.server_name, "version": self.server_version},
                },
            )
        if method == "tools/list":
            return self._result(req_id, {"tools": TOOL_DEFINITIONS})
        if method == "tools/call":
            return self._handle_tool_call(req_id, params)

        return self._error(req_id, -32601, f"Method not found: {method}")

    def _handle_tool_call(self, req_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        tool_name = params.get("name")
        args = params.get("arguments") or {}
        try:
            if tool_name == "memory_store":
                record = self.memory.store_memory(
                    content=args["content"],
                    session_id=args.get("session_id", "default"),
                    project=args.get("project", "default"),
                    tier=args.get("tier", "episodic"),
                    pinned=bool(args.get("pinned", False)),
                    expires_at=args.get("expires_at"),
                    scrub_secrets_enabled=bool(args.get("scrub_secrets", True)),
                    role=args.get("role", "user"),
                    strategy=args.get("strategy"),
                    timestamp=args.get("timestamp"),
                    tags=args.get("tags"),
                    source_refs=args.get("source_refs"),
                    metadata=args.get("metadata"),
                )
                return self._tool_ok(req_id, record.model_dump())

            if tool_name == "memory_search":
                hits = self.memory.search_memory(
                    args["query"],
                    strategy=args.get("strategy"),
                    limit=int(args.get("limit", 5)),
                    session_id=args.get("session_id"),
                    project=args.get("project"),
                    temporal_bucket=args.get("temporal_bucket"),
                )
                return self._tool_ok(req_id, [hit.model_dump() for hit in hits])

            if tool_name == "memory_get":
                record = self.memory.get_memory(args["memory_id"])
                return self._tool_ok(req_id, record.model_dump())

            if tool_name == "memory_delete":
                deleted = self.memory.delete_memory(args["memory_id"])
                return self._tool_ok(req_id, {"memory_id": args["memory_id"], "deleted": deleted})

            if tool_name == "memory_briefing":
                return self._tool_ok(
                    req_id,
                    self.memory.briefing(
                        project=args.get("project", "default"),
                        session_id=args.get("session_id"),
                        recent_limit=int(args.get("recent_limit", 8)),
                        settled_first=bool(args.get("settled_first", True)),
                    ),
                )

            if tool_name == "memory_feedback":
                return self._tool_ok(
                    req_id,
                    self.memory.submit_feedback(
                        args["memory_id"],
                        args["signal"],
                        reason=args.get("reason", ""),
                    ),
                )

            if tool_name == "memory_forget_sweep":
                return self._tool_ok(
                    req_id,
                    self.memory.forget_sweep(
                        dry_run=bool(args.get("dry_run", True)),
                        project=args.get("project"),
                        cold_threshold=args.get("cold_threshold"),
                        lambda_decay=args.get("lambda_decay"),
                    ),
                )

            if tool_name == "memory_restore":
                restored = self.memory.restore_memory(args["memory_id"])
                return self._tool_ok(req_id, restored.model_dump())

            if tool_name == "memory_lint":
                return self._tool_ok(req_id, self.memory.lint(project=args.get("project")))

            if tool_name == "memory_handoff_begin":
                record = self.memory.begin_handoff(HandoffCreateRequest(**args))
                return self._tool_ok(req_id, record.model_dump())

            if tool_name == "memory_handoff_list":
                items = self.memory.list_handoffs(
                    project=args.get("project"),
                    status=args.get("status", "pending"),
                    limit=int(args.get("limit", 20)),
                )
                return self._tool_ok(req_id, [item.model_dump() for item in items])

            if tool_name == "memory_handoff_accept":
                record = self.memory.accept_handoff(HandoffAcceptRequest(**args))
                return self._tool_ok(req_id, record.model_dump())

            if tool_name == "memory_handoff_cancel":
                record = self.memory.cancel_handoff(args["handoff_id"])
                return self._tool_ok(req_id, record.model_dump())

            if tool_name == "memory_message_send":
                msg = self.memory.send_message(AgentMessageSendRequest(**args))
                return self._tool_ok(req_id, msg.model_dump())

            if tool_name == "memory_message_list":
                msgs = self.memory.list_messages(
                    project=args.get("project", "default"),
                    box=args.get("box", "inbox"),
                    status=args.get("status", "pending"),
                    agent=args.get("agent"),
                    limit=int(args.get("limit", 50)),
                )
                return self._tool_ok(req_id, [m.model_dump() for m in msgs])

            if tool_name == "memory_message_pop":
                msg = self.memory.pop_message(AgentMessagePopRequest(**args))
                return self._tool_ok(req_id, msg.model_dump() if msg else {"claimed": False, "message": None})

            if tool_name == "memory_message_cancel":
                return self._tool_ok(req_id, self.memory.cancel_message(AgentMessageCancelRequest(**args)))

            if tool_name == "memory_write_page":
                page = self.memory.write_wiki_page(WikiPageWriteRequest(**args))
                return self._tool_ok(req_id, page.model_dump())

            if tool_name == "memory_patch_page":
                page = self.memory.patch_wiki_page(WikiPagePatchRequest(**args))
                return self._tool_ok(req_id, page.model_dump())

            if tool_name == "memory_read_page":
                return self._tool_ok(
                    req_id,
                    self.memory.read_wiki_page(
                        args["path"], include_related=bool(args.get("include_related", True))
                    ),
                )

            if tool_name == "memory_list_pages":
                if args.get("query"):
                    return self._tool_ok(
                        req_id,
                        self.memory.search_wiki_pages(
                            args["query"],
                            project=args.get("project"),
                            limit=int(args.get("limit", 50)),
                        ),
                    )
                pages = self.memory.list_wiki_pages(
                    project=args.get("project"),
                    kind=args.get("kind"),
                    include_deleted=bool(args.get("include_deleted", False)),
                    limit=int(args.get("limit", 50)),
                )
                return self._tool_ok(req_id, [p.model_dump() for p in pages])

            if tool_name == "memory_delete_page":
                return self._tool_ok(req_id, self.memory.delete_wiki_page(args["path"]))

            if tool_name == "memory_restore_page":
                restored = self.memory.restore_wiki_page(args["path"], version=args.get("version"))
                return self._tool_ok(req_id, restored.model_dump())

            return self._error(req_id, -32602, f"Unknown tool: {tool_name}")
        except KeyError as exc:
            return self._tool_error(req_id, f"Missing required argument: {exc}")
        except FileNotFoundError as exc:
            return self._tool_error(req_id, str(exc))
        except Exception as exc:
            return self._tool_error(req_id, str(exc))

    def _result(self, req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def _error(self, req_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}

    def _tool_ok(self, req_id: Any, payload: Any) -> dict[str, Any]:
        return self._result(
            req_id,
            {
                "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
                "structuredContent": payload,
                "isError": False,
            },
        )

    def _tool_error(self, req_id: Any, message: str) -> dict[str, Any]:
        return self._result(
            req_id,
            {
                "content": [{"type": "text", "text": message}],
                "isError": True,
            },
        )
