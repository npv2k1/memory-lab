# Hướng dẫn tích hợp Memory Lab vào LLM & AI Agent

Memory Lab không khóa chặt vào bất kỳ nhà cung cấp LLM nào. Bạn có thể tích hợp bộ nhớ dài hạn và RAG tài liệu của Memory Lab vào **Claude, GPT, Gemini, Ollama, vLLM, Cursor, Windsurf, LangGraph, PydanticAI** hoặc ứng dụng Python bất kỳ theo **3 cách**:

| Cách tích hợp | Phù hợp khi | Giao thức |
|---|---|---|
| **1. MCP (Model Context Protocol)** | Gắn trực tiếp vào Claude Desktop, Cursor, Windsurf, coding agents không cần viết code | JSON-RPC 2.0 qua `stdio` hoặc `POST /mcp` |
| **2. HTTP REST API (RAG + Tool Calling)** | Xây dựng chatbot/agent với OpenAI, Gemini, Claude, Ollama, vLLM ở bất kỳ ngôn ngữ nào | REST (`/search`, `/memory/*`) |
| **3. Python SDK nội bộ (In-Process)** | Nhúng trực tiếp vào app Python (LangGraph, CrewAI, FastAPI) không cần chạy server riêng | Import `MemoryService` / `DocumentPipeline` |

---

## Cách 1: Tích hợp qua MCP (Khuyên dùng cho Agent & IDE)

Memory Lab triển khai sẵn MCP server chuẩn JSON-RPC 2.0 với 4 tool:

- `memory_store`: lưu raw interaction trace và index theo chiến lược (`hybrid`, `zero_token`, `graph`, `vector`).
- `memory_search`: tìm kiếm ký ức (hỗ trợ lọc `strategy`, `session_id`, `temporal_bucket`).
- `memory_get`: đọc chi tiết một ký ức theo `memory_id`.
- `memory_delete`: xóa ký ức khỏi các chỉ mục hoạt động.

### 1A. Chạy qua `stdio` (Claude Desktop, Cursor, Windsurf)

**Chế độ gọn nhẹ không cần Docker / không tốn phí Embedding (`zero_token` hoặc `graph`):**

```json
{
  "mcpServers": {
    "memory-lab": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/ABSOLUTE/PATH/TO/memory-lab",
        "python",
        "-m",
        "memory_lab.mcp"
      ],
      "env": {
        "DATA_DIR": "/ABSOLUTE/PATH/TO/memory-lab/data",
        "QDRANT_URL": "path:/ABSOLUTE/PATH/TO/memory-lab/data/qdrant",
        "DEFAULT_MEMORY_STRATEGY": "zero_token"
      }
    }
  }
}
```

**Chế độ đầy đủ `hybrid` (Dense + BM25 + Graph + Temporal):**

Thêm `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, `EMBEDDING_API_KEY` (hoặc trỏ `QDRANT_URL=http://localhost:6333` nếu chạy Qdrant bằng Docker) và đặt `"DEFAULT_MEMORY_STRATEGY": "hybrid"`.

> `QDRANT_URL` hỗ trợ 3 dạng:
> - `http://localhost:6333` — kết nối Qdrant server (Docker Compose).
> - `path:/duong/dan/qdrant` — Qdrant nhúng lưu thẳng ra đĩa, **không cần Docker**.
> - `:memory:` — chạy trên RAM (phù hợp kiểm thử nhanh).

### 1B. Gọi MCP qua HTTP (`POST /mcp`)

Khi API đang chạy tại `http://localhost:8000` (hoặc qua Nginx `http://localhost:4321/api/mcp`):

```bash
# 1. Lấy danh sách tools
curl -s http://localhost:8000/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'

# 2. Lưu ký ức
curl -s http://localhost:8000/mcp \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "memory_store",
      "arguments": {
        "content": "Khách hàng ưu tiên triển khai on-premise bằng Docker Compose trên Ubuntu 24.04.",
        "session_id": "project-alpha",
        "strategy": "hybrid",
        "tags": ["deployment", "preference"]
      }
    }
  }'

# 3. Tìm kiếm ký ức
curl -s http://localhost:8000/mcp \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
      "name": "memory_search",
      "arguments": {
        "query": "yêu cầu triển khai on-premise",
        "strategy": "hybrid",
        "limit": 5
      }
    }
  }'
```

---

## Cách 2: Tích hợp qua REST API vào vòng lặp LLM (OpenAI / Gemini / Claude / Ollama)

Có hai mẫu tích hợp phổ biến nhất:

### 2A. Vòng lặp tự động `Retrieve → Prompt → Generate → Write-back`

Phù hợp cho chatbot hoặc trợ lý cá nhân cần tự động nhớ ngữ cảnh qua nhiều phiên và trả lời có trích dẫn chứng cứ từ cả **tài liệu** (`/search`) lẫn **bộ nhớ hội thoại** (`/memory/search`).

```python
import os
import httpx

MEMORY_LAB_URL = os.getenv("MEMORY_LAB_URL", "http://localhost:8000")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")  # hoặc http://localhost:11434/v1 cho Ollama
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def chat_with_memory(user_message: str, session_id: str = "default") -> str:
    with httpx.Client(timeout=60) as client:
        # 1. Truy xuất tài liệu (hybrid = Dense + BM25 RRF; đổi thành "sparse" nếu không dùng embedding)
        doc_hits = client.get(
            f"{MEMORY_LAB_URL}/search",
            params={"q": user_message, "mode": "hybrid", "limit": 3},
        ).json()

        # 2. Truy xuất bộ nhớ dài hạn của agent
        mem_hits = client.post(
            f"{MEMORY_LAB_URL}/memory/search",
            json={"query": user_message, "strategy": "hybrid", "limit": 5},
        ).json()

        # 3. Đóng gói ngữ cảnh giữ nguyên provenance (block_ids, timestamp, score)
        doc_context = "\n".join(
            f"- [doc:{h['chunk']['document_id'][:8]} blocks:{','.join(h['chunk']['block_ids'])}] {h['chunk']['text']}"
            for h in doc_hits
        ) or "(Không có tài liệu liên quan)"

        mem_context = "\n".join(
            f"- [{h['record']['timestamp'][:10]} | session:{h['record']['session_id']} | score:{h['score']:.3f}] {h['record']['content']}"
            for h in mem_hits
        ) or "(Chưa có ký ức liên quan)"

        system_prompt = (
            "Bạn là trợ lý AI sử dụng Memory Lab. Chỉ trả lời dựa trên bằng chứng cung cấp bên dưới; "
            "nếu không đủ thông tin hãy nói rõ.\n\n"
            f"<documents>\n{doc_context}\n</documents>\n\n"
            f"<agent_memory>\n{mem_context}\n</agent_memory>"
        )

        # 4. Gọi LLM chuẩn OpenAI Chat Completions
        headers = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
        llm_resp = client.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers=headers,
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            },
        )
        llm_resp.raise_for_status()
        answer = llm_resp.json()["choices"][0]["message"]["content"]

        # 5. Ghi lại tương tác vào Memory Lab (chọn lọc hoặc ghi nguyên cặp Q&A)
        client.post(
            f"{MEMORY_LAB_URL}/memory/store",
            json={
                "content": f"User: {user_message}\nAssistant: {answer}",
                "session_id": session_id,
                "role": "assistant",
                "strategy": "hybrid",
            },
        ).raise_for_status()

        return answer
```

### 2B. Cho phép LLM chủ động gọi Tool (Function Calling)

Thay vì tự động lưu mọi tin nhắn, bạn đưa thẳng schema từ `TOOL_DEFINITIONS` (trong `memory_lab.mcp.server`) cho LLM dưới dạng tools để model **tự quyết định** khi nào cần gọi `memory_search` hoặc `memory_store`:

```python
import json
import httpx
from memory_lab.mcp.server import TOOL_DEFINITIONS

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["inputSchema"],
        },
    }
    for t in TOOL_DEFINITIONS
]


def execute_memory_tool(tool_name: str, arguments: dict, memory_lab_url: str = "http://localhost:8000") -> str:
    """Chuyển tiếp trực tiếp tool call của LLM sang endpoint POST /mcp của Memory Lab."""
    resp = httpx.post(
        f"{memory_lab_url}/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        },
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()["result"]
    return json.dumps(result.get("structuredContent", result["content"]), ensure_ascii=False)
```

Cách chuyển tiếp sang `POST /mcp` này đảm bảo schema tool cho OpenAI / Gemini / Claude luôn đồng bộ 100% với MCP server mà không phải viết lại logic từng endpoint.

---

## Cách 3: Nhúng trực tiếp bằng Python SDK (In-Process, không cần HTTP server)

Phù hợp khi viết agent bằng **LangGraph**, **PydanticAI**, **CrewAI** hoặc dịch vụ Python đơn khối:

```python
from pathlib import Path
from memory_lab.memory.service import MemoryService
from memory_lab.retrieval.embedding import OpenAIEmbeddings
from memory_lab.storage.filesystem.store import FileStore
from memory_lab.storage.qdrant.index import QdrantIndex, create_qdrant_client

store = FileStore(Path("./data"))
embedding = OpenAIEmbeddings(base_url="http://localhost:11434/v1", api_key="", model="nomic-embed-text")
index = QdrantIndex(create_qdrant_client("path:./data/qdrant"), collection="agent")

memory = MemoryService(
    store=store,
    embedding=embedding,
    index=index,
    embedding_identity="ollama:nomic-embed-text",
    default_strategy="hybrid",  # hoặc "zero_token" nếu không chạy embedding server
)

# Lưu ký ức
rec = memory.store_memory(
    content="Quyết định kiến trúc 2026-11-25: dùng Qdrant và BM25 RRF cho tìm kiếm tài liệu.",
    session_id="arch-review",
    tags=["architecture"],
)

# Truy xuất ký ức trước khi gọi LLM
hits = memory.search_memory("quyết định kiến trúc Qdrant", limit=5, temporal_bucket="2026-11")
for hit in hits:
    print(hit.score, hit.score_breakdown, hit.record.content)
```

---

## Chọn chiến lược truy xuất phù hợp cho LLM

| Nhu cầu của Agent | `strategy` (Memory) | `mode` (Documents) | Yêu cầu hạ tầng |
|---|---|---|---|
| Chạy 0đ API / local siêu nhanh / offline | `zero_token` | `sparse` | Chỉ cần filesystem |
| Suy luận liên kết thực thể (`A -> B -> C`) | `graph` | `sparse` hoặc `hybrid` | Filesystem (hoặc Neo4j tùy chọn) |
| Tìm theo ý nghĩa câu chữ (paraphrase) | `vector` | `dense` | Embedding server + Qdrant |
| **Tổng hợp tốt nhất cho production (Khuyên dùng)** | **`hybrid`** | **`hybrid`** | Embedding + Qdrant (tự hạ cấp mượt nếu offline) |

## Thực hành tốt (Best Practices)

1. **Tách `session_id` theo người dùng / dự án**: truyền `session_id` khi `store` và `search` nếu muốn cô lập ngữ cảnh từng user hoặc từng workspace.
2. **Không lưu toàn bộ hội thoại nhiễu vào memory**: với chat dài, ưu tiên dùng chế độ **Tool Calling (Cách 2B)** hoặc chỉ lưu các quyết định, sở thích, sự kiện quan trọng để giữ chất lượng truy xuất cao.
3. **Khai thác `temporal_bucket`**: khi người dùng hỏi theo mốc thời gian (*"tháng 11/2026 mình bàn gì?"*), truyền `temporal_bucket="2026-11"` để thu hẹp không gian tìm kiếm và tăng độ chính xác.
4. **Đo lại sau mỗi lần đổi prompt/model/chiến lược**: tạo dataset benchmark trên tab **Experimentation & Comparison** (`POST /experiments/run`) để kiểm tra `Recall@K`, `MRR`, `nDCG@K` trước khi đưa cấu hình mới vào agent.
5. **Bảo mật**: Memory Lab mặc định không có lớp authentication (thiết kế cho mạng nội bộ / loopback). Khi nối với LLM ở môi trường production, luôn đặt sau reverse proxy có xác thực (Bearer token / mTLS / VPN) và không mở thẳng cổng `8000` ra Internet.
