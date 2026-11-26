import json
from typing import Any
from memory_lab.memory.service import MemoryService

SUPPORTED_PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")
DEFAULT_PROTOCOL_VERSION = "2025-03-26"

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "memory_store",
        "description": "Store an agent interaction trace and index it using the selected memory strategy.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Interaction or memory text content."},
                "session_id": {"type": "string", "default": "default"},
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
        "description": "Search stored memories using vector, graph, zero_token, or hybrid retrieval.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "strategy": {"type": "string", "enum": ["vector", "graph", "zero_token", "hybrid"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
                "session_id": {"type": "string"},
                "temporal_bucket": {"type": "string", "description": "Optional YYYY, YYYY-MM, or YYYY-MM-DD filter."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_get",
        "description": "Retrieve a stored memory record by ID.",
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
        "description": "Delete a memory record by ID and remove it from active indexes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Unique memory record ID."},
            },
            "required": ["memory_id"],
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
                    temporal_bucket=args.get("temporal_bucket"),
                )
                return self._tool_ok(req_id, [hit.model_dump() for hit in hits])

            if tool_name == "memory_get":
                record = self.memory.get_memory(args["memory_id"])
                return self._tool_ok(req_id, record.model_dump())

            if tool_name == "memory_delete":
                deleted = self.memory.delete_memory(args["memory_id"])
                return self._tool_ok(req_id, {"memory_id": args["memory_id"], "deleted": deleted})

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
        # MCP requires structuredContent to be a JSON object; clients (e.g. the
        # TypeScript SDK) reject arrays, so list payloads are wrapped.
        structured = payload if isinstance(payload, dict) else {"results": payload}
        return self._result(
            req_id,
            {
                "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
                "structuredContent": structured,
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
