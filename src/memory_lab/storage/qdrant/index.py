from qdrant_client import QdrantClient, models
from memory_lab.core.models import Chunk, Evidence


class QdrantIndex:
    def __init__(self, client: QdrantClient, collection: str):
        self.client, self.collection = client, collection

    def name(self, namespace: str) -> str:
        return f"{self.collection}_{namespace}"

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]], namespace: str) -> None:
        if len(chunks) != len(vectors) or not vectors:
            raise ValueError("Expected one embedding per chunk")
        name = self.name(namespace)
        if not self.client.collection_exists(name):
            self.client.create_collection(name, vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE))
        for offset in range(0, len(chunks), 64):
            points = [models.PointStruct(id=chunk.id, vector=vector, payload=chunk.model_dump())
                for chunk, vector in zip(chunks[offset:offset + 64], vectors[offset:offset + 64], strict=True)]
            self.client.upsert(name, points=points, wait=True)

    def search(self, vector: list[float], limit: int, namespace: str) -> list[Evidence]:
        name = self.name(namespace)
        if not self.client.collection_exists(name):
            return []
        hits = self.client.query_points(name, query=vector, limit=limit, with_payload=True).points
        result = []
        for hit in hits:
            chunk = Chunk.model_validate(hit.payload)
            result.append(Evidence(chunk=chunk, score=hit.score,
                source_url=f"/documents/{chunk.document_id}/versions/{chunk.version}/source"))
        return result
