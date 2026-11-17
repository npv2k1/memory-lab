import httpx


class OpenAIEmbeddings:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.model:
            raise ValueError("Set EMBEDDING_MODEL before processing/searching")
        vectors = []
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        with httpx.Client(timeout=120, headers=headers) as client:
            for offset in range(0, len(texts), 32):
                batch = texts[offset:offset + 32]
                response = client.post(f"{self.base_url}/embeddings", json={"model": self.model, "input": batch})
                response.raise_for_status()
                items = sorted(response.json()["data"], key=lambda row: row["index"])
                if [item["index"] for item in items] != list(range(len(batch))):
                    raise ValueError("Invalid embedding batch response")
                vectors.extend(item["embedding"] for item in items)
        if any(not vector for vector in vectors) or len({len(v) for v in vectors}) > 1:
            raise ValueError("Invalid embedding dimensions")
        return vectors
