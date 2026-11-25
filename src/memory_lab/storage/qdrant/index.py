from typing import Any
from qdrant_client import QdrantClient, models
from memory_lab.core.models import Chunk, Evidence


def create_qdrant_client(qdrant_url: str) -> QdrantClient:
    """Create a QdrantClient from an HTTP URL, ':memory:', or 'path:<dir>' for embedded local storage."""
    if qdrant_url == ":memory:":
        return QdrantClient(location=":memory:")
    if qdrant_url.startswith("path:"):
        return QdrantClient(path=qdrant_url.removeprefix("path:"))
    return QdrantClient(url=qdrant_url)


class QdrantIndex:
    def __init__(self, client: QdrantClient, collection: str):
        self.client, self.collection = client, collection

    def name(self, namespace: str) -> str:
        return f"{self.collection}_{namespace}"

    def _build_filter(self, filters: dict[str, str] | None) -> models.Filter | None:
        if not filters:
            return None
        conditions = [
            models.FieldCondition(key=field, match=models.MatchValue(value=value))
            for field, value in filters.items()
            if value
        ]
        return models.Filter(must=conditions) if conditions else None

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]], namespace: str) -> None:
        if len(chunks) != len(vectors) or not vectors:
            raise ValueError("Expected one embedding per chunk")
        name = self.name(namespace)
        if not self.client.collection_exists(name):
            self.client.create_collection(
                name,
                vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
            )
        for offset in range(0, len(chunks), 64):
            points = [
                models.PointStruct(id=chunk.id, vector=vector, payload=chunk.model_dump())
                for chunk, vector in zip(chunks[offset:offset + 64], vectors[offset:offset + 64], strict=True)
            ]
            self.client.upsert(name, points=points, wait=True)

    def search(
        self, vector: list[float], limit: int, namespace: str, filters: dict[str, str] | None = None
    ) -> list[Evidence]:
        name = self.name(namespace)
        if not self.client.collection_exists(name):
            return []
        query_filter = self._build_filter(filters)
        hits = self.client.query_points(
            name, query=vector, limit=limit, query_filter=query_filter, with_payload=True
        ).points
        result = []
        for hit in hits:
            chunk = Chunk.model_validate(hit.payload)
            result.append(
                Evidence(
                    chunk=chunk,
                    score=hit.score,
                    source_url=f"/documents/{chunk.document_id}/versions/{chunk.version}/source",
                    score_breakdown={"dense": round(float(hit.score), 6)},
                )
            )
        return result

    def upsert_points(
        self, point_ids: list[str], vectors: list[list[float]], payloads: list[dict[str, Any]], namespace: str
    ) -> None:
        if not (len(point_ids) == len(vectors) == len(payloads)) or not vectors:
            raise ValueError("Expected matching IDs, vectors, and payloads")
        name = self.name(namespace)
        if not self.client.collection_exists(name):
            self.client.create_collection(
                name,
                vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
            )
        points = [
            models.PointStruct(id=pid, vector=vec, payload=payload)
            for pid, vec, payload in zip(point_ids, vectors, payloads, strict=True)
        ]
        self.client.upsert(name, points=points, wait=True)

    def search_points(
        self, vector: list[float], limit: int, namespace: str, filters: dict[str, str] | None = None
    ) -> list[tuple[str, float, dict[str, Any]]]:
        name = self.name(namespace)
        if not self.client.collection_exists(name):
            return []
        query_filter = self._build_filter(filters)
        hits = self.client.query_points(
            name, query=vector, limit=limit, query_filter=query_filter, with_payload=True
        ).points
        return [(str(hit.id), float(hit.score), dict(hit.payload or {})) for hit in hits]

    def delete_points(self, point_ids: list[str], namespace: str) -> None:
        if not point_ids:
            return
        name = self.name(namespace)
        if not self.client.collection_exists(name):
            return
        self.client.delete(
            collection_name=name,
            points_selector=models.PointIdsList(points=point_ids),
            wait=True,
        )
