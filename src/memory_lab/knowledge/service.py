from collections import Counter, defaultdict
import time
from typing import Any, Literal
from memory_lab.core.models import (
    Chunk,
    Document,
    EnrichmentArtifact,
    MemoryStrategyName,
    SearchMode,
)
from memory_lab.documents.enrichment.extractor import normalize_name
from memory_lab.documents.pipeline.service import DocumentPipeline
from memory_lab.memory.service import MemoryService
from memory_lab.storage.filesystem.store import FileStore

TableViewName = Literal["entities", "relations", "chunks", "memories", "timeline"]
KnowledgeSourceFilter = Literal["all", "documents", "memory"]


class KnowledgeService:
    """Unified knowledge aggregation, tabular views, interactive graph, and human search across Documents + Memory."""

    def __init__(self, store: FileStore, pipeline: DocumentPipeline, memory: MemoryService):
        self.store = store
        self.pipeline = pipeline
        self.memory = memory

    def _documents_map(self) -> dict[tuple[str, str], Document]:
        docs: dict[tuple[str, str], Document] = {}
        for raw in self.store.documents():
            doc = Document.model_validate(raw)
            docs[(doc.id, doc.version)] = doc
        return docs

    def _completed_runs(self) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for path in sorted(self.store.root.glob("runs/*/manifest.json")):
            manifest = self.store.get(str(path.relative_to(self.store.root)))
            runs.append(manifest)
        return runs

    def _load_enrichments(self) -> list[EnrichmentArtifact]:
        artifacts: list[EnrichmentArtifact] = []
        for path in sorted(self.store.root.glob("runs/*/enriched.json")):
            try:
                artifacts.append(EnrichmentArtifact.model_validate(self.store.get(str(path.relative_to(self.store.root)))))
            except Exception:
                continue
        return artifacts

    def _load_chunks(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        seen: set[str] = set()
        for path in sorted(self.store.root.glob("runs/*/chunks.json")):
            try:
                rows = self.store.get(str(path.relative_to(self.store.root)))
                for row in rows:
                    chunk = Chunk.model_validate(row)
                    if chunk.id not in seen:
                        seen.add(chunk.id)
                        chunks.append(chunk)
            except Exception:
                continue
        return chunks

    def _aggregate_graph_data(
        self,
        *,
        source: KnowledgeSourceFilter = "all",
        include_cooccurrence: bool = True,
    ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        docs_map = self._documents_map()
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str, str, str]] = set()

        def ensure_node(
            norm_id: str,
            name: str,
            etype: str,
            src_kind: str,
            evidence_label: str,
        ) -> dict[str, Any]:
            node = nodes.get(norm_id)
            if node is None:
                node = {
                    "id": norm_id,
                    "name": name,
                    "type": etype or "concept",
                    "mentions": 0,
                    "document_count": 0,
                    "memory_count": 0,
                    "sources": set(),
                    "evidence_refs": set(),
                }
                nodes[norm_id] = node
            elif node["type"] == "concept" and etype and etype != "concept":
                node["type"] = etype
            node["mentions"] += 1
            node["sources"].add(src_kind)
            if src_kind == "document":
                node["document_count"] += 1
            else:
                node["memory_count"] += 1
            if evidence_label and len(node["evidence_refs"]) < 6:
                node["evidence_refs"].add(evidence_label)
            return node

        # 1. Aggregate Document Enrichments
        if source in ("all", "documents"):
            for artifact in self._load_enrichments():
                doc = docs_map.get((artifact.document_id, artifact.version))
                label = doc.filename if doc else f"doc:{artifact.document_id[:8]}"
                for entity in artifact.entities:
                    ensure_node(entity.normalized_name, entity.name, entity.type, "document", label)
                for rel in artifact.relations:
                    if not include_cooccurrence and rel.predicate == "co_occurs_with":
                        continue
                    subj = normalize_name(rel.subject)
                    obj = normalize_name(rel.object)
                    if not subj or not obj or subj == obj:
                        continue
                    if subj not in nodes:
                        ensure_node(subj, rel.subject, "concept", "document", label)
                    if obj not in nodes:
                        ensure_node(obj, rel.object, "concept", "document", label)
                    edge_key = (subj, rel.predicate, obj, "document")
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges.append(
                            {
                                "id": f"doc:{len(edges)}",
                                "source": subj,
                                "target": obj,
                                "predicate": rel.predicate,
                                "confidence": round(float(rel.confidence), 3),
                                "source_type": "document",
                                "evidence_id": rel.evidence_id or artifact.pipeline_run_id,
                                "source_label": label,
                            }
                        )

        # 2. Aggregate Agent Memory Records
        if source in ("all", "memory"):
            for record in self.memory.list_records(limit=500):
                label = f"memory:{record.session_id} ({record.timestamp[:10]})"
                for entity in record.entities:
                    ensure_node(entity.normalized_name, entity.name, entity.type, "memory", label)
                for rel in record.relations:
                    if not include_cooccurrence and rel.predicate == "co_occurs_with":
                        continue
                    subj = normalize_name(rel.subject)
                    obj = normalize_name(rel.object)
                    if not subj or not obj or subj == obj:
                        continue
                    if subj not in nodes:
                        ensure_node(subj, rel.subject, "concept", "memory", label)
                    if obj not in nodes:
                        ensure_node(obj, rel.object, "concept", "memory", label)
                    edge_key = (subj, rel.predicate, obj, "memory")
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges.append(
                            {
                                "id": f"mem:{len(edges)}",
                                "source": subj,
                                "target": obj,
                                "predicate": rel.predicate,
                                "confidence": round(float(rel.confidence), 3),
                                "source_type": "memory",
                                "evidence_id": record.id,
                                "source_label": label,
                            }
                        )

        return nodes, edges

    def overview(self) -> dict[str, Any]:
        docs = self.store.documents()
        runs = self._completed_runs()
        chunks = self._load_chunks()
        records = self.memory.list_records(limit=500)
        nodes, edges = self._aggregate_graph_data(source="all", include_cooccurrence=True)

        type_counts = Counter(node["type"] for node in nodes.values())
        degree = Counter()
        for e in edges:
            degree[e["source"]] += 1
            degree[e["target"]] += 1

        top_entities = sorted(
            (
                {
                    "id": nid,
                    "name": node["name"],
                    "type": node["type"],
                    "mentions": node["mentions"],
                    "degree": degree.get(nid, 0),
                    "sources": sorted(node["sources"]),
                }
                for nid, node in nodes.items()
            ),
            key=lambda item: (-item["mentions"], -item["degree"], item["name"]),
        )[:12]

        temporal_set: set[str] = set()
        for art in self._load_enrichments():
            temporal_set.update(t.normalized for t in art.temporal)
        for rec in records:
            temporal_set.update(t.normalized for t in rec.temporal)
            temporal_set.add(rec.timestamp[:10])

        datasets = list(self.store.root.glob("experiments/datasets/*.json"))
        reports = list(self.store.root.glob("experiments/reports/*.json"))

        return {
            "documents_count": len(docs),
            "runs_count": len(runs),
            "chunks_count": len(chunks),
            "memory_records_count": len(records),
            "entities_count": len(nodes),
            "relations_count": len(edges),
            "temporal_anchors_count": len(temporal_set),
            "datasets_count": len(datasets),
            "reports_count": len(reports),
            "entity_type_breakdown": dict(type_counts),
            "top_entities": top_entities,
        }

    def graph(
        self,
        *,
        entity: str | None = None,
        source: KnowledgeSourceFilter = "all",
        include_cooccurrence: bool = True,
        max_nodes: int = 80,
    ) -> dict[str, Any]:
        nodes_map, all_edges = self._aggregate_graph_data(
            source=source, include_cooccurrence=include_cooccurrence
        )

        degree = Counter()
        for edge in all_edges:
            degree[edge["source"]] += 1
            degree[edge["target"]] += 1

        if entity and entity.strip():
            target = normalize_name(entity)
            matched_ids = {
                nid for nid in nodes_map if nid == target or target in nid or nid in target
            }
            sub_edges = [
                e for e in all_edges if e["source"] in matched_ids or e["target"] in matched_ids
            ]
            keep_ids = set(matched_ids)
            for e in sub_edges:
                keep_ids.add(e["source"])
                keep_ids.add(e["target"])
            selected_ids = sorted(
                keep_ids,
                key=lambda nid: (-degree.get(nid, 0), -nodes_map[nid]["mentions"], nid),
            )[:max_nodes]
        else:
            selected_ids = sorted(
                nodes_map.keys(),
                key=lambda nid: (-degree.get(nid, 0), -nodes_map[nid]["mentions"], nid),
            )[:max_nodes]

        selected_set = set(selected_ids)
        filtered_edges = [
            e for e in all_edges if e["source"] in selected_set and e["target"] in selected_set
        ]

        out_nodes = [
            {
                "id": nid,
                "name": nodes_map[nid]["name"],
                "type": nodes_map[nid]["type"],
                "mentions": nodes_map[nid]["mentions"],
                "degree": degree.get(nid, 0),
                "document_count": nodes_map[nid]["document_count"],
                "memory_count": nodes_map[nid]["memory_count"],
                "sources": sorted(nodes_map[nid]["sources"]),
                "evidence_refs": sorted(nodes_map[nid]["evidence_refs"]),
            }
            for nid in selected_ids
        ]

        return {
            "focus_entity": normalize_name(entity) if entity else None,
            "source_filter": source,
            "nodes": out_nodes,
            "edges": filtered_edges,
            "stats": {
                "node_count": len(out_nodes),
                "edge_count": len(filtered_edges),
                "total_nodes_available": len(nodes_map),
                "total_edges_available": len(all_edges),
                "document_nodes": sum(1 for n in out_nodes if "document" in n["sources"]),
                "memory_nodes": sum(1 for n in out_nodes if "memory" in n["sources"]),
            },
        }

    def table(
        self,
        *,
        view: TableViewName = "entities",
        q: str | None = None,
        entity_type: str | None = None,
        source: KnowledgeSourceFilter = "all",
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        needle = q.strip().lower() if q else ""
        rows: list[dict[str, Any]] = []
        columns: list[str] = []

        if view == "entities":
            columns = ["name", "type", "mentions", "degree", "sources", "evidence_refs"]
            nodes_map, edges = self._aggregate_graph_data(source=source, include_cooccurrence=True)
            degree = Counter()
            for e in edges:
                degree[e["source"]] += 1
                degree[e["target"]] += 1
            for nid, node in nodes_map.items():
                if entity_type and node["type"] != entity_type:
                    continue
                row = {
                    "id": nid,
                    "name": node["name"],
                    "type": node["type"],
                    "mentions": node["mentions"],
                    "degree": degree.get(nid, 0),
                    "document_count": node["document_count"],
                    "memory_count": node["memory_count"],
                    "sources": sorted(node["sources"]),
                    "evidence_refs": sorted(node["evidence_refs"]),
                }
                if needle and needle not in f"{row['name']} {row['type']} {' '.join(row['evidence_refs'])}".lower():
                    continue
                rows.append(row)
            rows.sort(key=lambda r: (-r["mentions"], -r["degree"], r["name"]))

        elif view == "relations":
            columns = ["subject", "predicate", "object", "confidence", "source_type", "source_label"]
            nodes_map, edges = self._aggregate_graph_data(source=source, include_cooccurrence=True)
            for edge in edges:
                subj_name = nodes_map.get(edge["source"], {}).get("name", edge["source"])
                obj_name = nodes_map.get(edge["target"], {}).get("name", edge["target"])
                row = {
                    "id": edge["id"],
                    "subject": subj_name,
                    "subject_id": edge["source"],
                    "predicate": edge["predicate"],
                    "object": obj_name,
                    "object_id": edge["target"],
                    "confidence": edge["confidence"],
                    "source_type": edge["source_type"],
                    "source_label": edge["source_label"],
                    "evidence_id": edge["evidence_id"],
                }
                hay = f"{row['subject']} {row['predicate']} {row['object']} {row['source_label']}".lower()
                if needle and needle not in hay:
                    continue
                rows.append(row)
            rows.sort(key=lambda r: (-r["confidence"], r["predicate"], r["subject"]))

        elif view == "chunks":
            columns = ["filename", "block_ids", "entities", "temporal", "preview", "source_url"]
            docs_map = self._documents_map()
            enriched_by_chunk: dict[str, dict[str, Any]] = {}
            for art in self._load_enrichments():
                for ec in art.chunks:
                    enriched_by_chunk[ec.chunk_id] = {
                        "entities": [e.name for e in ec.entities],
                        "temporal": [t.normalized for t in ec.temporal],
                    }
            for chunk in self._load_chunks():
                doc = docs_map.get((chunk.document_id, chunk.version))
                fname = doc.filename if doc else f"{chunk.document_id[:10]}"
                meta = enriched_by_chunk.get(chunk.id, {"entities": [], "temporal": []})
                row = {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "version": chunk.version,
                    "pipeline_run_id": chunk.pipeline_run_id,
                    "filename": fname,
                    "block_ids": chunk.block_ids,
                    "entities": meta["entities"],
                    "temporal": meta["temporal"],
                    "preview": chunk.text[:240],
                    "text": chunk.text,
                    "source_url": f"/documents/{chunk.document_id}/versions/{chunk.version}/source",
                }
                hay = f"{fname} {chunk.text} {' '.join(meta['entities'])}".lower()
                if needle and needle not in hay:
                    continue
                rows.append(row)

        elif view == "memories":
            columns = ["timestamp", "strategy", "session_id", "entities", "tags", "summary"]
            for rec in self.memory.list_records(limit=500):
                row = {
                    "id": rec.id,
                    "timestamp": rec.timestamp,
                    "strategy": rec.strategy,
                    "session_id": rec.session_id,
                    "role": rec.role,
                    "entities": [e.name for e in rec.entities],
                    "tags": rec.tags,
                    "summary": rec.summary or rec.content[:200],
                    "content": rec.content,
                }
                hay = f"{rec.content} {rec.session_id} {rec.strategy} {' '.join(row['entities'])} {' '.join(rec.tags)}".lower()
                if needle and needle not in hay:
                    continue
                rows.append(row)

        elif view == "timeline":
            columns = ["date", "granularity", "source_type", "label", "entities", "preview"]
            docs_map = self._documents_map()
            if source in ("all", "documents"):
                for art in self._load_enrichments():
                    doc = docs_map.get((art.document_id, art.version))
                    fname = doc.filename if doc else art.document_id[:10]
                    for anchor in art.temporal:
                        row = {
                            "date": anchor.normalized,
                            "granularity": anchor.granularity,
                            "source_type": "document",
                            "label": fname,
                            "entities": [e.name for e in art.entities[:5]],
                            "preview": f"Mốc '{anchor.raw}' trích xuất từ {fname}",
                            "ref_id": art.pipeline_run_id,
                        }
                        if needle and needle not in f"{row['date']} {row['label']} {' '.join(row['entities'])}".lower():
                            continue
                        rows.append(row)
            if source in ("all", "memory"):
                for rec in self.memory.list_records(limit=500):
                    row = {
                        "date": rec.timestamp[:10],
                        "granularity": "day",
                        "source_type": "memory",
                        "label": f"{rec.session_id} · {rec.strategy}",
                        "entities": [e.name for e in rec.entities[:5]],
                        "preview": rec.summary or rec.content[:180],
                        "ref_id": rec.id,
                    }
                    if needle and needle not in f"{row['date']} {row['label']} {rec.content}".lower():
                        continue
                    rows.append(row)
            rows.sort(key=lambda r: (r["date"], r["source_type"]), reverse=True)
        else:
            raise ValueError(f"Unsupported knowledge table view: {view}")

        total = len(rows)
        sliced = rows[offset : offset + limit]
        return {
            "view": view,
            "source": source,
            "query": q or "",
            "total": total,
            "limit": limit,
            "offset": offset,
            "columns": columns,
            "rows": sliced,
        }

    def search(
        self,
        query: str,
        *,
        scope: KnowledgeSourceFilter = "all",
        doc_mode: SearchMode = "hybrid",
        memory_strategy: MemoryStrategyName = "hybrid",
        temporal_bucket: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        docs_map = self._documents_map()

        document_hits: list[dict[str, Any]] = []
        if scope in ("all", "documents"):
            try:
                hits = self.pipeline.search(query, limit, mode=doc_mode)
            except Exception:
                hits = self.pipeline.search(query, limit, mode="sparse")
            for hit in hits:
                doc = docs_map.get((hit.chunk.document_id, hit.chunk.version))
                item = hit.model_dump()
                item["filename"] = doc.filename if doc else hit.chunk.document_id[:10]
                document_hits.append(item)

        memory_hits: list[dict[str, Any]] = []
        if scope in ("all", "memory"):
            try:
                mhits = self.memory.search_memory(
                    query,
                    strategy=memory_strategy,
                    limit=limit,
                    temporal_bucket=temporal_bucket,
                )
            except Exception:
                mhits = self.memory.search_memory(
                    query,
                    strategy="zero_token",
                    limit=limit,
                    temporal_bucket=temporal_bucket,
                )
            memory_hits = [hit.model_dump() for hit in mhits]

        # Also surface matching entities & relations from the unified graph
        q_terms = {t.lower() for t in query.split() if len(t) >= 2}
        nodes_map, edges = self._aggregate_graph_data(source=scope, include_cooccurrence=True)
        matched_entities = []
        for nid, node in nodes_map.items():
            name_low = node["name"].lower()
            if query.lower() in name_low or any(t in nid or t in name_low for t in q_terms):
                matched_entities.append(
                    {
                        "id": nid,
                        "name": node["name"],
                        "type": node["type"],
                        "mentions": node["mentions"],
                        "sources": sorted(node["sources"]),
                    }
                )
        matched_entities.sort(key=lambda e: (-e["mentions"], e["name"]))

        matched_entity_ids = {e["id"] for e in matched_entities[:12]}
        matched_relations = []
        for edge in edges:
            if edge["source"] in matched_entity_ids or edge["target"] in matched_entity_ids:
                matched_relations.append(
                    {
                        "subject": nodes_map.get(edge["source"], {}).get("name", edge["source"]),
                        "subject_id": edge["source"],
                        "predicate": edge["predicate"],
                        "object": nodes_map.get(edge["target"], {}).get("name", edge["target"]),
                        "object_id": edge["target"],
                        "confidence": edge["confidence"],
                        "source_type": edge["source_type"],
                        "source_label": edge["source_label"],
                    }
                )
        matched_relations = matched_relations[:12]

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return {
            "query": query,
            "scope": scope,
            "doc_mode": doc_mode,
            "memory_strategy": memory_strategy,
            "latency_ms": elapsed_ms,
            "total_hits": len(document_hits) + len(memory_hits),
            "documents": document_hits,
            "memories": memory_hits,
            "matched_entities": matched_entities[:12],
            "matched_relations": matched_relations,
        }
