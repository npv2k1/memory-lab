from collections import Counter, defaultdict
from typing import Any
import httpx
from memory_lab.core.models import MemoryRecord
from memory_lab.documents.enrichment.extractor import normalize_name
from memory_lab.storage.filesystem.store import FileStore


class FileGraphStore:
    """Filesystem-backed property graph for entities, relations, and memory record provenance."""

    def __init__(self, store: FileStore):
        self.store = store
        self._key = "memory/graph/graph.json"

    def _load(self) -> dict[str, Any]:
        if not self.store.path(self._key).exists():
            return {"nodes": {}, "edges": []}
        return self.store.get(self._key)

    def _save(self, data: dict[str, Any]) -> None:
        self.store.put(self._key, data)

    def upsert_record(self, record: MemoryRecord) -> None:
        with self.store.lock("memory_graph"):
            data = self._load()
            self._remove_record_unlocked(data, record.id)
            nodes: dict[str, dict[str, Any]] = data["nodes"]
            edges: list[dict[str, Any]] = data["edges"]

            for entity in record.entities:
                key = entity.normalized_name
                node = nodes.setdefault(
                    key,
                    {"id": key, "name": entity.name, "type": entity.type, "record_ids": []},
                )
                if record.id not in node["record_ids"]:
                    node["record_ids"].append(record.id)

            for relation in record.relations:
                for endpoint in (relation.subject, relation.object):
                    key = normalize_name(endpoint)
                    node = nodes.setdefault(
                        key,
                        {"id": key, "name": endpoint, "type": "concept", "record_ids": []},
                    )
                    if record.id not in node["record_ids"]:
                        node["record_ids"].append(record.id)
                edges.append(
                    {
                        "subject": normalize_name(relation.subject),
                        "predicate": relation.predicate,
                        "object": normalize_name(relation.object),
                        "record_id": record.id,
                        "evidence_id": relation.evidence_id or record.id,
                        "confidence": relation.confidence,
                    }
                )
            self._save(data)

    def _remove_record_unlocked(self, data: dict[str, Any], record_id: str) -> None:
        data["edges"] = [edge for edge in data["edges"] if edge.get("record_id") != record_id]
        to_delete: list[str] = []
        for node_id, node in data["nodes"].items():
            node["record_ids"] = [rid for rid in node.get("record_ids", []) if rid != record_id]
            if not node["record_ids"]:
                to_delete.append(node_id)
        for node_id in to_delete:
            del data["nodes"][node_id]

    def delete_record(self, record_id: str) -> None:
        with self.store.lock("memory_graph"):
            data = self._load()
            self._remove_record_unlocked(data, record_id)
            self._save(data)

    def query_by_entities(self, entities: list[str], limit: int = 10) -> list[tuple[str, float]]:
        data = self._load()
        nodes: dict[str, dict[str, Any]] = data["nodes"]
        edges: list[dict[str, Any]] = data["edges"]
        if not nodes or not entities:
            return []

        query_norms = {normalize_name(e) for e in entities if e.strip()}
        matched_nodes: set[str] = set()
        for node_id in nodes:
            if node_id in query_norms or any(q in node_id or node_id in q for q in query_norms if len(q) >= 3):
                matched_nodes.add(node_id)

        if not matched_nodes:
            return []

        scores: dict[str, float] = defaultdict(float)
        # 1-hop direct entity matches
        for node_id in matched_nodes:
            for record_id in nodes[node_id].get("record_ids", []):
                scores[record_id] += 1.0

        # 2-hop graph neighbor expansion through edges
        neighbor_nodes: set[str] = set()
        for edge in edges:
            subj, obj = edge["subject"], edge["object"]
            weight = 0.65 if edge.get("predicate") != "co_occurs_with" else 0.35
            if subj in matched_nodes:
                neighbor_nodes.add(obj)
                scores[edge["record_id"]] += weight * float(edge.get("confidence", 1.0))
            if obj in matched_nodes:
                neighbor_nodes.add(subj)
                scores[edge["record_id"]] += weight * float(edge.get("confidence", 1.0))

        for neighbor_id in neighbor_nodes - matched_nodes:
            neighbor = nodes.get(neighbor_id)
            if not neighbor:
                continue
            for record_id in neighbor.get("record_ids", []):
                scores[record_id] += 0.3

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [(rid, round(score, 6)) for rid, score in ranked[:limit]]

    def snapshot(self, entity: str | None = None) -> dict[str, Any]:
        data = self._load()
        nodes = data["nodes"]
        edges = data["edges"]
        if entity:
            target = normalize_name(entity)
            matched_edges = [e for e in edges if e["subject"] == target or e["object"] == target]
            related_ids = {target}
            for e in matched_edges:
                related_ids.add(e["subject"])
                related_ids.add(e["object"])
            sub_nodes = [nodes[nid] for nid in sorted(related_ids) if nid in nodes]
            return {
                "entity": target,
                "nodes": sub_nodes,
                "edges": matched_edges,
                "node_count": len(sub_nodes),
                "edge_count": len(matched_edges),
            }

        degree = Counter()
        for e in edges:
            degree[e["subject"]] += 1
            degree[e["object"]] += 1
        node_list = [
            {**node, "degree": degree.get(nid, 0)}
            for nid, node in sorted(nodes.items())
        ]
        return {
            "nodes": node_list,
            "edges": edges,
            "node_count": len(node_list),
            "edge_count": len(edges),
        }


class Neo4jGraphStore:
    """Optional Neo4j HTTP Query API adapter backed by a FileGraphStore mirror."""

    def __init__(
        self,
        url: str,
        user: str,
        password: str,
        database: str = "neo4j",
        *,
        mirror: FileGraphStore | None = None,
    ):
        self.url = url.rstrip("/")
        self.auth = (user, password)
        self.database = database
        self.mirror = mirror

    def _cypher(self, statements: list[dict[str, Any]]) -> dict[str, Any]:
        endpoint = f"{self.url}/db/{self.database}/tx/commit"
        with httpx.Client(timeout=15, auth=self.auth) as client:
            response = client.post(endpoint, json={"statements": statements})
            response.raise_for_status()
            payload = response.json()
            if payload.get("errors"):
                raise RuntimeError(f"Neo4j error: {payload['errors']}")
            return payload

    def upsert_record(self, record: MemoryRecord) -> None:
        if self.mirror:
            self.mirror.upsert_record(record)
        stmts: list[dict[str, Any]] = [
            {
                "statement": "MERGE (m:MemoryRecord {id: $id}) SET m.session_id = $session_id, m.timestamp = $ts",
                "parameters": {"id": record.id, "session_id": record.session_id, "ts": record.timestamp},
            }
        ]
        for entity in record.entities:
            stmts.append(
                {
                    "statement": (
                        "MERGE (e:Entity {id: $eid}) SET e.name = $name, e.type = $etype "
                        "WITH e MATCH (m:MemoryRecord {id: $mid}) MERGE (m)-[:MENTIONS]->(e)"
                    ),
                    "parameters": {
                        "eid": entity.normalized_name,
                        "name": entity.name,
                        "etype": entity.type,
                        "mid": record.id,
                    },
                }
            )
        for rel in record.relations:
            stmts.append(
                {
                    "statement": (
                        "MERGE (s:Entity {id: $subj}) MERGE (o:Entity {id: $obj}) "
                        "MERGE (s)-[r:RELATED {predicate: $pred, record_id: $mid}]->(o) "
                        "SET r.confidence = $conf"
                    ),
                    "parameters": {
                        "subj": normalize_name(rel.subject),
                        "obj": normalize_name(rel.object),
                        "pred": rel.predicate,
                        "mid": record.id,
                        "conf": rel.confidence,
                    },
                }
            )
        self._cypher(stmts)

    def delete_record(self, record_id: str) -> None:
        if self.mirror:
            self.mirror.delete_record(record_id)
        self._cypher(
            [
                {
                    "statement": (
                        "MATCH (m:MemoryRecord {id: $id}) DETACH DELETE m "
                        "WITH $id AS rid MATCH ()-[r:RELATED {record_id: rid}]-() DELETE r"
                    ),
                    "parameters": {"id": record_id},
                }
            ]
        )

    def query_by_entities(self, entities: list[str], limit: int = 10) -> list[tuple[str, float]]:
        norms = [normalize_name(e) for e in entities if e.strip()]
        if not norms:
            return []
        try:
            payload = self._cypher(
                [
                    {
                        "statement": (
                            "MATCH (m:MemoryRecord)-[:MENTIONS]->(e:Entity) "
                            "WHERE e.id IN $entities "
                            "OPTIONAL MATCH (e)-[:RELATED]-(n:Entity)<-[:MENTIONS]-(m2:MemoryRecord) "
                            "RETURN m.id AS id, count(DISTINCT e) * 1.0 AS score "
                            "ORDER BY score DESC LIMIT $limit"
                        ),
                        "parameters": {"entities": norms, "limit": limit},
                    }
                ]
            )
            rows = payload["results"][0]["data"]
            return [(row["row"][0], float(row["row"][1])) for row in rows]
        except Exception:
            if self.mirror:
                return self.mirror.query_by_entities(entities, limit=limit)
            raise

    def snapshot(self, entity: str | None = None) -> dict[str, Any]:
        if self.mirror:
            return self.mirror.snapshot(entity=entity)
        return {"nodes": [], "edges": [], "node_count": 0, "edge_count": 0}
