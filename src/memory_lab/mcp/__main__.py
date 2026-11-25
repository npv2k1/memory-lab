import json
import sys
from qdrant_client import QdrantClient
from memory_lab.core.config import Settings
from memory_lab.mcp.server import MCPServer
from memory_lab.memory.service import MemoryService
from memory_lab.retrieval.embedding import OpenAIEmbeddings
from memory_lab.storage.filesystem.store import FileStore
from memory_lab.storage.graph.store import FileGraphStore, Neo4jGraphStore
from memory_lab.storage.qdrant.index import QdrantIndex


def build_default_mcp_server(settings: Settings | None = None) -> MCPServer:
    settings = settings or Settings()
    store = FileStore(settings.data_dir)
    embedding = OpenAIEmbeddings(settings.embedding_base_url, settings.embedding_api_key, settings.embedding_model)
    index = QdrantIndex(QdrantClient(url=settings.qdrant_url), settings.qdrant_collection)
    file_graph = FileGraphStore(store)
    graph = (
        Neo4jGraphStore(
            settings.neo4j_url, settings.neo4j_user, settings.neo4j_password, settings.neo4j_database, mirror=file_graph
        )
        if settings.neo4j_url
        else file_graph
    )
    memory = MemoryService(
        store,
        embedding,
        index,
        f"{settings.embedding_base_url}:{settings.embedding_model}",
        graph=graph,
        default_strategy=settings.default_memory_strategy,  # type: ignore[arg-type]
    )
    return MCPServer(memory)


def run_stdio(server: MCPServer | None = None) -> None:
    mcp = server or build_default_mcp_server()
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}}
        else:
            response = mcp.handle_message(message)
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_stdio()
