from collections import Counter, defaultdict
from pathlib import Path
import re
import time
from typing import Any, Literal
from memory_lab.core.models import (
    Chunk,
    Document,
    EnrichmentArtifact,
    MemoryStrategyName,
    NormalizedDocument,
    SearchMode,
)
from memory_lab.documents.enrichment.extractor import (
    PREDICATE_LABELS_VI,
    fold_ascii,
    normalize_name,
)
from memory_lab.documents.enrichment.vi_tokenizer import (
    extract_evidence_sentence,
    extract_topic_phrases,
    is_single_syllable_vi_noise,
)
from memory_lab.documents.pipeline.service import DocumentPipeline
from memory_lab.knowledge.topics import build_topic_map
from memory_lab.memory.service import MemoryService
from memory_lab.storage.filesystem.store import FileStore

TableViewName = Literal["entities", "relations", "chunks", "memories", "timeline"]
KnowledgeSourceFilter = Literal["all", "documents", "memory"]


class KnowledgeService:
    """Unified knowledge aggregation, Topic Map, Ego Graph, tabular views, and Ask search across Documents + Memory."""

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

    def _all_run_manifests(self) -> list[tuple[dict[str, Any], float]]:
        runs: list[tuple[dict[str, Any], float]] = []
        for path in sorted(self.store.root.glob("runs/*/manifest.json")):
            try:
                manifest = self.store.get(str(path.relative_to(self.store.root)))
                runs.append((manifest, path.stat().st_mtime))
            except Exception:
                continue
        return runs

    def _latest_run_by_doc(self) -> dict[tuple[str, str], dict[str, Any]]:
        """Map (document_id, version) -> most recent pipeline run manifest (preferring active/completed)."""
        best: dict[tuple[str, str], tuple[int, str, float, dict[str, Any]]] = {}
        status_rank = {"processing": 3, "completed": 2, "invalidated": 1, "failed": 0}
        for manifest, mtime in self._all_run_manifests():
            doc_id = manifest.get("document_id")
            ver = manifest.get("version")
            if not doc_id or not ver:
                continue
            ts = str(manifest.get("completed_at") or manifest.get("updated_at") or manifest.get("started_at") or "")
            rank = status_rank.get(str(manifest.get("status", "")), 0)
            key = (doc_id, ver)
            prev = best.get(key)
            cand = (rank, ts, mtime, manifest)
            if prev is None or (cand[1], cand[2], cand[0]) > (prev[1], prev[2], prev[0]):
                best[key] = cand
        return {k: v[3] for k, v in best.items()}

    def _completed_runs(self) -> list[dict[str, Any]]:
        return [m for m, _mtime in self._all_run_manifests()]

    def _active_run_ids(self) -> set[str]:
        """Latest completed run ID per (document_id, version) so multi-run re-processing never duplicates stats."""
        latest_completed: dict[tuple[str, str], tuple[str, float, str]] = {}
        for manifest, mtime in self._all_run_manifests():
            if manifest.get("status") != "completed":
                continue
            doc_id = manifest.get("document_id")
            ver = manifest.get("version")
            run_id = manifest.get("pipeline_run_id")
            if not doc_id or not ver or not run_id:
                continue
            ts = str(manifest.get("completed_at") or manifest.get("updated_at") or "")
            key = (doc_id, ver)
            prev = latest_completed.get(key)
            if prev is None or (ts, mtime, run_id) > prev:
                latest_completed[key] = (ts, mtime, run_id)
        return {v[2] for v in latest_completed.values()}

    def _load_enrichments(self, *, latest_only: bool = True) -> list[EnrichmentArtifact]:
        active_ids = self._active_run_ids() if latest_only else set()
        artifacts: list[EnrichmentArtifact] = []
        for path in sorted(self.store.root.glob("runs/*/enriched.json")):
            run_id = path.parent.name
            if latest_only and active_ids and run_id not in active_ids:
                continue
            try:
                artifacts.append(
                    EnrichmentArtifact.model_validate(self.store.get(str(path.relative_to(self.store.root))))
                )
            except Exception:
                continue
        return artifacts

    def _load_chunks(self, *, latest_only: bool = True) -> list[Chunk]:
        active_ids = self._active_run_ids() if latest_only else set()
        chunks: list[Chunk] = []
        seen: set[str] = set()
        for path in sorted(self.store.root.glob("runs/*/chunks.json")):
            run_id = path.parent.name
            if latest_only and active_ids and run_id not in active_ids:
                continue
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

    def _chunk_text_map(self) -> dict[str, Chunk]:
        return {c.id: c for c in self._load_chunks(latest_only=True)}

    def _block_meta_for_run(self, run_id: str) -> dict[str, dict[str, Any]]:
        norm_key = f"runs/{run_id}/normalized.json"
        if not self.store.path(norm_key).exists():
            return {}
        try:
            norm = NormalizedDocument.model_validate(self.store.get(norm_key))
            return {
                b.id: {
                    "heading": " › ".join(b.heading_hierarchy) if b.heading_hierarchy else "",
                    "page": b.page,
                    "type": b.type,
                }
                for b in norm.blocks
            }
        except Exception:
            return {}

    def documents_with_status(self) -> list[dict[str, Any]]:
        """Return all stored documents enriched with their latest pipeline run status, size, and extension."""
        latest_runs = self._latest_run_by_doc()
        enriched: list[dict[str, Any]] = []
        for raw in self.store.documents():
            doc = Document.model_validate(raw)
            raw_path = self.store.raw_path(doc)
            size_bytes = raw_path.stat().st_size if raw_path.exists() else 0
            mtime = raw_path.stat().st_mtime if raw_path.exists() else 0.0
            ext = Path(doc.filename).suffix.lstrip(".").lower() or "txt"
            run = latest_runs.get((doc.id, doc.version))
            latest_summary = None
            if run:
                cfg = run.get("config") or {}
                latest_summary = {
                    "pipeline_run_id": run.get("pipeline_run_id"),
                    "status": run.get("status"),
                    "progress": run.get("progress", 100 if run.get("status") == "completed" else 0),
                    "current_stage": run.get("current_stage") or run.get("status"),
                    "stages": run.get("stages") or {},
                    "chunk_count": run.get("chunk_count", 0),
                    "enrichment_summary": run.get("enrichment_summary"),
                    "parser": cfg.get("parser"),
                    "chunker": cfg.get("chunker"),
                    "updated_at": run.get("completed_at") or run.get("updated_at") or run.get("started_at"),
                    "error_type": run.get("error_type"),
                    "error_message": run.get("error_message"),
                }
            enriched.append(
                {
                    **doc.model_dump(),
                    "extension": ext,
                    "size_bytes": size_bytes,
                    "updated_ts": mtime,
                    "latest_run": latest_summary,
                }
            )
        enriched.sort(key=lambda item: (item["updated_ts"], item["filename"]), reverse=True)
        return enriched

    def document_detail(self, identity: str, version: str) -> dict[str, Any]:
        """Build a clean human-readable document reading view (TOC, formatted chunks, top entities, main topics)."""
        doc = self.store.document(identity, version)
        latest_runs = self._latest_run_by_doc()
        run = latest_runs.get((doc.id, doc.version))

        toc: list[dict[str, Any]] = []
        chunks_out: list[dict[str, Any]] = []
        top_entities: list[dict[str, Any]] = []
        topics: list[str] = []

        if run and run.get("pipeline_run_id"):
            run_id = str(run["pipeline_run_id"])
            block_meta = self._block_meta_for_run(run_id)

            chunks_key = f"runs/{run_id}/chunks.json"
            raw_chunks = (
                [Chunk.model_validate(r) for r in self.store.get(chunks_key)]
                if self.store.path(chunks_key).exists()
                else []
            )
            enriched_key = f"runs/{run_id}/enriched.json"
            enriched = (
                EnrichmentArtifact.model_validate(self.store.get(enriched_key))
                if self.store.path(enriched_key).exists()
                else None
            )
            ec_by_id = {ec.chunk_id: ec for ec in (enriched.chunks if enriched else [])}

            seen_headings: set[str] = set()
            ent_counter: Counter[str] = Counter()
            ent_meta: dict[str, dict[str, str]] = {}
            phrase_counter: Counter[str] = Counter()

            md_heading_re = re.compile(r"^\s*#{1,6}\s+(.+?)\s*#*$", re.MULTILINE)
            num_heading_re = re.compile(r"^\s*(?:(?:Chương|Phần|Mục)\s+[IVXLC\d]+[:.\s-]+|\d+(?:\.\d+){0,2}\.\s+)([A-ZÀ-Ỹ][^\n]{2,80})$", re.MULTILINE)

            for idx, chunk in enumerate(raw_chunks):
                heading = ""
                page = None
                for bid in chunk.block_ids:
                    bm = block_meta.get(bid)
                    if bm:
                        if not heading and bm.get("heading"):
                            heading = bm["heading"]
                        if page is None and bm.get("page") is not None:
                            page = bm["page"]
                found_headings: list[str] = []
                if heading:
                    found_headings.append(heading)
                else:
                    for m in md_heading_re.finditer(chunk.text):
                        h = m.group(1).strip()
                        if h and len(h) <= 120:
                            found_headings.append(h)
                    if not found_headings:
                        for m in num_heading_re.finditer(chunk.text):
                            h = m.group(0).strip()
                            if h and len(h) <= 120:
                                found_headings.append(h)
                    if found_headings:
                        heading = found_headings[0]
                for h in found_headings:
                    if h not in seen_headings:
                        seen_headings.add(h)
                        toc.append({"heading": h, "chunk_index": idx, "page": page})

                ec = ec_by_id.get(chunk.id)
                chunk_entities: list[dict[str, str]] = []
                if ec:
                    for e in ec.entities:
                        chunk_entities.append({"id": e.normalized_name, "name": e.name, "type": e.type})
                        ent_counter[e.normalized_name] += 1
                        ent_meta.setdefault(e.normalized_name, {"id": e.normalized_name, "name": e.name, "type": e.type})

                phrase_counter.update(extract_topic_phrases(chunk.text))
                chunks_out.append(
                    {
                        "id": chunk.id,
                        "index": idx,
                        "text": chunk.text,
                        "heading": heading,
                        "page": page,
                        "block_ids": chunk.block_ids,
                        "entities": chunk_entities[:8],
                    }
                )

            top_entities = [
                {**ent_meta[eid], "mentions": cnt}
                for eid, cnt in ent_counter.most_common(12)
                if not is_single_syllable_vi_noise(ent_meta[eid]["name"])
            ]
            # Top meaningful document topic phrases
            for phrase, _cnt in phrase_counter.most_common(20):
                if len(phrase.split()) >= 2 or not any(ord(c) > 127 for c in phrase):
                    clean = phrase[0].upper() + phrase[1:]
                    if not any(clean.lower() in t.lower() or t.lower() in clean.lower() for t in topics):
                        topics.append(clean)
                    if len(topics) >= 6:
                        break

        raw_path = self.store.raw_path(doc)
        return {
            "document": {
                **doc.model_dump(),
                "extension": Path(doc.filename).suffix.lstrip(".").lower() or "txt",
                "size_bytes": raw_path.stat().st_size if raw_path.exists() else 0,
                "source_url": f"/documents/{doc.id}/versions/{doc.version}/source",
            },
            "latest_run": run,
            "toc": toc,
            "chunks": chunks_out,
            "top_entities": top_entities,
            "topics": topics,
        }

    def _aggregate_graph_data(
        self,
        *,
        source: KnowledgeSourceFilter = "all",
        include_cooccurrence: bool = True,
    ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        docs_map = self._documents_map()
        chunk_map = self._chunk_text_map()
        nodes: dict[str, dict[str, Any]] = {}
        edges_map: dict[tuple[str, str, str], dict[str, Any]] = {}

        def ensure_node(
            norm_id: str,
            name: str,
            etype: str,
            src_kind: str,
            evidence_label: str,
            evidence_sentence: str = "",
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
                    "evidence_sentences": [],
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
            if evidence_sentence and len(node["evidence_sentences"]) < 3 and evidence_sentence not in node["evidence_sentences"]:
                node["evidence_sentences"].append(evidence_sentence)
            return node

        def record_edge(
            subj: str,
            subj_name: str,
            predicate: str,
            obj: str,
            obj_name: str,
            confidence: float,
            src_kind: str,
            evidence_id: str,
            source_label: str,
            evidence_sentence: str = "",
        ) -> None:
            if not subj or not obj or subj == obj:
                return
            if subj not in nodes:
                ensure_node(subj, subj_name, "concept", src_kind, source_label, evidence_sentence)
            if obj not in nodes:
                ensure_node(obj, obj_name, "concept", src_kind, source_label, evidence_sentence)
            # Canonicalize undirected co-occurrence edge key
            ekey = (min(subj, obj), predicate, max(subj, obj)) if predicate == "co_occurs_with" else (subj, predicate, obj)
            existing = edges_map.get(ekey)
            if existing is None:
                edges_map[ekey] = {
                    "id": f"e:{len(edges_map)}",
                    "source": ekey[0],
                    "target": ekey[2],
                    "predicate": predicate,
                    "predicate_label": PREDICATE_LABELS_VI.get(predicate, predicate),
                    "confidence": round(float(confidence), 3),
                    "weight": 1,
                    "source_type": src_kind,
                    "evidence_id": evidence_id,
                    "source_label": source_label,
                    "evidence_sentence": evidence_sentence,
                }
            else:
                existing["weight"] += 1
                existing["confidence"] = max(existing["confidence"], round(float(confidence), 3))
                if not existing["evidence_sentence"] and evidence_sentence:
                    existing["evidence_sentence"] = evidence_sentence

        # 1. Aggregate Document Enrichments (chunk-level mentions so mention counts & quoted sentences are real!)
        if source in ("all", "documents"):
            for artifact in self._load_enrichments(latest_only=True):
                doc = docs_map.get((artifact.document_id, artifact.version))
                label = doc.filename if doc else f"doc:{artifact.document_id[:8]}"
                if artifact.chunks:
                    for ec in artifact.chunks:
                        chunk_obj = chunk_map.get(ec.chunk_id)
                        chunk_text = chunk_obj.text if chunk_obj else ""
                        for entity in ec.entities:
                            sent = extract_evidence_sentence(chunk_text, entity.name) if chunk_text else ""
                            ensure_node(entity.normalized_name, entity.name, entity.type, "document", label, sent)
                        for rel in ec.relations:
                            if not include_cooccurrence and rel.predicate == "co_occurs_with":
                                continue
                            subj = normalize_name(rel.subject)
                            obj = normalize_name(rel.object)
                            sent = extract_evidence_sentence(chunk_text, rel.subject) if chunk_text else ""
                            record_edge(
                                subj, rel.subject, rel.predicate, obj, rel.object,
                                rel.confidence, "document", ec.chunk_id, label, sent,
                            )
                else:
                    for entity in artifact.entities:
                        ensure_node(entity.normalized_name, entity.name, entity.type, "document", label)
                    for rel in artifact.relations:
                        if not include_cooccurrence and rel.predicate == "co_occurs_with":
                            continue
                        record_edge(
                            normalize_name(rel.subject), rel.subject, rel.predicate,
                            normalize_name(rel.object), rel.object,
                            rel.confidence, "document", rel.evidence_id or artifact.pipeline_run_id, label,
                        )

        # 2. Aggregate Agent Memory Records
        if source in ("all", "memory"):
            for record in self.memory.list_records(limit=500):
                label = f"Ký ức · {record.timestamp[:10]}"
                for entity in record.entities:
                    sent = extract_evidence_sentence(record.content, entity.name)
                    ensure_node(entity.normalized_name, entity.name, entity.type, "memory", label, sent)
                for rel in record.relations:
                    if not include_cooccurrence and rel.predicate == "co_occurs_with":
                        continue
                    sent = extract_evidence_sentence(record.content, rel.subject)
                    record_edge(
                        normalize_name(rel.subject), rel.subject, rel.predicate,
                        normalize_name(rel.object), rel.object,
                        rel.confidence, "memory", record.id, label, sent,
                    )

        return nodes, list(edges_map.values())

    def topics(
        self,
        *,
        source: KnowledgeSourceFilter = "all",
        n_topics: int | None = None,
    ) -> dict[str, Any]:
        """Build c-TF-IDF Topic Map over active document chunks and/or agent memories."""
        docs_map = self._documents_map()
        items: list[dict[str, Any]] = []

        if source in ("all", "documents"):
            enriched_by_chunk: dict[str, list[dict[str, str]]] = {}
            for art in self._load_enrichments(latest_only=True):
                for ec in art.chunks:
                    enriched_by_chunk[ec.chunk_id] = [
                        {"id": e.normalized_name, "name": e.name, "type": e.type}
                        for e in ec.entities
                        if not is_single_syllable_vi_noise(e.name)
                    ]

            block_meta_cache: dict[str, dict[str, dict[str, Any]]] = {}
            for chunk in self._load_chunks(latest_only=True):
                doc = docs_map.get((chunk.document_id, chunk.version))
                fname = doc.filename if doc else f"doc:{chunk.document_id[:8]}"
                bmeta = block_meta_cache.get(chunk.pipeline_run_id)
                if bmeta is None:
                    bmeta = self._block_meta_for_run(chunk.pipeline_run_id)
                    block_meta_cache[chunk.pipeline_run_id] = bmeta
                heading = ""
                for bid in chunk.block_ids:
                    if bid in bmeta and bmeta[bid].get("heading"):
                        heading = bmeta[bid]["heading"]
                        break
                items.append(
                    {
                        "id": chunk.id,
                        "source_type": "document",
                        "source_label": fname,
                        "document_id": chunk.document_id,
                        "version": chunk.version,
                        "heading": heading,
                        "text": chunk.text,
                        "entities": enriched_by_chunk.get(chunk.id, []),
                    }
                )

        if source in ("all", "memory"):
            for rec in self.memory.list_records(limit=500):
                items.append(
                    {
                        "id": rec.id,
                        "source_type": "memory",
                        "source_label": f"Ký ức ({rec.session_id})",
                        "document_id": None,
                        "version": None,
                        "heading": "",
                        "text": rec.content,
                        "entities": [
                            {"id": e.normalized_name, "name": e.name, "type": e.type}
                            for e in rec.entities
                            if not is_single_syllable_vi_noise(e.name)
                        ],
                    }
                )

        return build_topic_map(items, n_topics=n_topics)

    def entity_neighbors(
        self,
        entity_id: str,
        *,
        depth: int = 1,
        limit: int = 15,
        include_cooccurrence: bool = False,
        min_cooccurrence: int = 3,
        entity_type: str | None = None,
        source: KnowledgeSourceFilter = "all",
    ) -> dict[str, Any]:
        """Build a focused Ego Graph centered on a single entity (1-hop up to `limit`, optional depth=2)."""
        nodes_map, all_edges = self._aggregate_graph_data(source=source, include_cooccurrence=True)
        target_norm = normalize_name(entity_id)
        target_folded = fold_ascii(target_norm)

        # Resolve exact or folded center entity
        center_id: str | None = None
        if target_norm in nodes_map:
            center_id = target_norm
        else:
            for nid in nodes_map:
                if fold_ascii(nid) == target_folded:
                    center_id = nid
                    break
            if center_id is None:
                # Substring fallback picking highest-mention match
                cands = [
                    nid for nid in nodes_map
                    if target_norm in nid or target_folded in fold_ascii(nid)
                ]
                if cands:
                    cands.sort(key=lambda nid: (-nodes_map[nid]["mentions"], nid))
                    center_id = cands[0]

        if center_id is None:
            raise FileNotFoundError(f"Entity not found: {entity_id}")

        # Filter edges: explicit labeled edges always included; co_occurs_with hidden by default unless weight >= min_cooccurrence
        def edge_allowed(e: dict[str, Any]) -> bool:
            if e["predicate"] != "co_occurs_with":
                return True
            if include_cooccurrence:
                return int(e.get("weight", 1)) >= max(1, min_cooccurrence)
            return False

        usable_edges = [e for e in all_edges if edge_allowed(e)]

        def incident_edges(nid: str) -> list[dict[str, Any]]:
            inc = [e for e in usable_edges if e["source"] == nid or e["target"] == nid]
            inc.sort(
                key=lambda e: (
                    0 if e["predicate"] != "co_occurs_with" else 1,
                    -int(e.get("weight", 1)),
                    -float(e.get("confidence", 0.0)),
                )
            )
            return inc

        # 1-hop neighbors
        hop1_edges = incident_edges(center_id)
        hop1_neighbors: list[str] = []
        selected_edges: list[dict[str, Any]] = []
        seen_edge_ids: set[str] = set()

        for edge in hop1_edges:
            other = edge["target"] if edge["source"] == center_id else edge["source"]
            other_node = nodes_map.get(other)
            if not other_node:
                continue
            if entity_type and other_node["type"] != entity_type:
                continue
            if other not in hop1_neighbors:
                if len(hop1_neighbors) >= limit:
                    continue
                hop1_neighbors.append(other)
            if edge["id"] not in seen_edge_ids:
                seen_edge_ids.add(edge["id"])
                selected_edges.append(edge)

        node_hops: dict[str, int] = {center_id: 0}
        for nid in hop1_neighbors:
            node_hops[nid] = 1

        # Optional 2-hop expansion (capped tightly so graph stays legible!)
        if depth >= 2 and hop1_neighbors:
            hop2_budget = max(6, min(12, limit))
            hop2_added = 0
            for n1 in hop1_neighbors[:6]:
                for edge in incident_edges(n1)[:4]:
                    other = edge["target"] if edge["source"] == n1 else edge["source"]
                    if other == center_id or other in node_hops:
                        if other in node_hops and edge["id"] not in seen_edge_ids:
                            seen_edge_ids.add(edge["id"])
                            selected_edges.append(edge)
                        continue
                    other_node = nodes_map.get(other)
                    if not other_node:
                        continue
                    if entity_type and other_node["type"] != entity_type:
                        continue
                    if hop2_added >= hop2_budget:
                        continue
                    node_hops[other] = 2
                    hop2_added += 1
                    if edge["id"] not in seen_edge_ids:
                        seen_edge_ids.add(edge["id"])
                        selected_edges.append(edge)

        # Degree count within ego subgraph
        sub_degree = Counter()
        for e in selected_edges:
            sub_degree[e["source"]] += 1
            sub_degree[e["target"]] += 1

        out_nodes = []
        for nid, hop in sorted(node_hops.items(), key=lambda pair: (pair[1], -nodes_map[pair[0]]["mentions"], pair[0])):
            n = nodes_map[nid]
            out_nodes.append(
                {
                    "id": nid,
                    "name": n["name"],
                    "type": n["type"],
                    "hop": hop,
                    "mentions": n["mentions"],
                    "degree": sub_degree.get(nid, 0),
                    "sources": sorted(n["sources"]),
                    "evidence_refs": sorted(n["evidence_refs"]),
                    "evidence_sentence": n["evidence_sentences"][0] if n["evidence_sentences"] else "",
                }
            )

        # Count hidden co-occurrence edges so UI can offer "Hiện thêm N liên kết cùng ngữ cảnh"
        hidden_cooccur = sum(
            1
            for e in all_edges
            if e["predicate"] == "co_occurs_with"
            and (e["source"] == center_id or e["target"] == center_id)
            and e["id"] not in seen_edge_ids
        )

        return {
            "center": out_nodes[0],
            "depth": depth,
            "nodes": out_nodes,
            "edges": selected_edges,
            "hidden_cooccurrence_count": hidden_cooccur,
            "stats": {
                "node_count": len(out_nodes),
                "edge_count": len(selected_edges),
                "explicit_edge_count": sum(1 for e in selected_edges if e["predicate"] != "co_occurs_with"),
            },
        }

    def overview(self) -> dict[str, Any]:
        docs = self.store.documents()
        runs = self._completed_runs()
        chunks = self._load_chunks(latest_only=True)
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
                if not is_single_syllable_vi_noise(node["name"])
            ),
            key=lambda item: (-item["mentions"], -item["degree"], item["name"]),
        )[:12]

        temporal_set: set[str] = set()
        for art in self._load_enrichments(latest_only=True):
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
                "evidence_sentence": (
                    nodes_map[nid]["evidence_sentences"][0] if nodes_map[nid]["evidence_sentences"] else ""
                ),
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
        min_mentions: int = 1,
        hide_single_syllable: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        needle = q.strip().lower() if q else ""
        rows: list[dict[str, Any]] = []
        columns: list[str] = []

        if view == "entities":
            columns = ["name", "type", "mentions", "degree", "sources", "evidence_sentence", "evidence_refs"]
            nodes_map, edges = self._aggregate_graph_data(source=source, include_cooccurrence=False)
            all_nodes_map, all_edges = self._aggregate_graph_data(source=source, include_cooccurrence=True)
            degree = Counter()
            for e in all_edges:
                degree[e["source"]] += 1
                degree[e["target"]] += 1
            explicit_degree = Counter()
            for e in edges:
                explicit_degree[e["source"]] += 1
                explicit_degree[e["target"]] += 1

            for nid, node in all_nodes_map.items():
                if entity_type and node["type"] != entity_type:
                    continue
                if node["mentions"] < min_mentions:
                    continue
                if hide_single_syllable and is_single_syllable_vi_noise(node["name"]):
                    continue
                ev_sentence = node["evidence_sentences"][0] if node["evidence_sentences"] else ""
                row = {
                    "id": nid,
                    "name": node["name"],
                    "type": node["type"],
                    "mentions": node["mentions"],
                    "degree": degree.get(nid, 0),
                    "explicit_degree": explicit_degree.get(nid, 0),
                    "document_count": node["document_count"],
                    "memory_count": node["memory_count"],
                    "sources": sorted(node["sources"]),
                    "evidence_sentence": ev_sentence,
                    "evidence_refs": sorted(node["evidence_refs"]),
                }
                hay = f"{row['name']} {row['type']} {ev_sentence} {' '.join(row['evidence_refs'])}".lower()
                if needle and needle not in hay:
                    continue
                rows.append(row)
            rows.sort(key=lambda r: (-r["mentions"], -r["explicit_degree"], -r["degree"], r["name"]))

        elif view == "relations":
            columns = ["subject", "predicate", "object", "weight", "confidence", "source_type", "source_label"]
            nodes_map, edges = self._aggregate_graph_data(source=source, include_cooccurrence=True)
            for edge in edges:
                subj_name = nodes_map.get(edge["source"], {}).get("name", edge["source"])
                obj_name = nodes_map.get(edge["target"], {}).get("name", edge["target"])
                row = {
                    "id": edge["id"],
                    "subject": subj_name,
                    "subject_id": edge["source"],
                    "predicate": edge["predicate"],
                    "predicate_label": edge.get("predicate_label", edge["predicate"]),
                    "object": obj_name,
                    "object_id": edge["target"],
                    "weight": edge.get("weight", 1),
                    "confidence": edge["confidence"],
                    "source_type": edge["source_type"],
                    "source_label": edge["source_label"],
                    "evidence_sentence": edge.get("evidence_sentence", ""),
                    "evidence_id": edge["evidence_id"],
                }
                hay = f"{row['subject']} {row['predicate']} {row['predicate_label']} {row['object']} {row['source_label']}".lower()
                if needle and needle not in hay:
                    continue
                rows.append(row)
            rows.sort(
                key=lambda r: (
                    0 if r["predicate"] != "co_occurs_with" else 1,
                    -r["weight"],
                    -r["confidence"],
                    r["subject"],
                )
            )

        elif view == "chunks":
            columns = ["filename", "block_ids", "entities", "temporal", "preview", "source_url"]
            docs_map = self._documents_map()
            enriched_by_chunk: dict[str, dict[str, Any]] = {}
            for art in self._load_enrichments(latest_only=True):
                for ec in art.chunks:
                    enriched_by_chunk[ec.chunk_id] = {
                        "entities": [e.name for e in ec.entities],
                        "temporal": [t.normalized for t in ec.temporal],
                    }
            for chunk in self._load_chunks(latest_only=True):
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
                for art in self._load_enrichments(latest_only=True):
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
        document_id: str | None = None,
        temporal_bucket: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        docs_map = self._documents_map()

        # Preload run chunks & enrichments lazily for context drawer (heading, page, adjacent chunks, entities)
        run_chunks_cache: dict[str, list[Chunk]] = {}
        run_block_meta_cache: dict[str, dict[str, dict[str, Any]]] = {}
        run_enriched_cache: dict[str, dict[str, list[dict[str, str]]]] = {}

        def _get_run_context(run_id: str):
            if run_id not in run_chunks_cache:
                ck = f"runs/{run_id}/chunks.json"
                run_chunks_cache[run_id] = (
                    [Chunk.model_validate(r) for r in self.store.get(ck)]
                    if self.store.path(ck).exists()
                    else []
                )
                run_block_meta_cache[run_id] = self._block_meta_for_run(run_id)
                ek = f"runs/{run_id}/enriched.json"
                ec_map: dict[str, list[dict[str, str]]] = {}
                if self.store.path(ek).exists():
                    try:
                        art = EnrichmentArtifact.model_validate(self.store.get(ek))
                        for ec in art.chunks:
                            ec_map[ec.chunk_id] = [
                                {"id": e.normalized_name, "name": e.name, "type": e.type}
                                for e in ec.entities
                                if not is_single_syllable_vi_noise(e.name)
                            ]
                    except Exception:
                        pass
                run_enriched_cache[run_id] = ec_map
            return run_chunks_cache[run_id], run_block_meta_cache[run_id], run_enriched_cache[run_id]

        document_hits: list[dict[str, Any]] = []
        if scope in ("all", "documents"):
            try:
                hits = self.pipeline.search(query, limit, document_id=document_id, mode=doc_mode)
            except Exception:
                hits = self.pipeline.search(query, limit, document_id=document_id, mode="sparse")
            for hit in hits:
                doc = docs_map.get((hit.chunk.document_id, hit.chunk.version))
                item = hit.model_dump()
                item["filename"] = doc.filename if doc else hit.chunk.document_id[:10]
                run_chunks, bmeta, ec_map = _get_run_context(hit.chunk.pipeline_run_id)
                heading = ""
                page = None
                for bid in hit.chunk.block_ids:
                    bm = bmeta.get(bid)
                    if bm:
                        if not heading and bm.get("heading"):
                            heading = bm["heading"]
                        if page is None and bm.get("page") is not None:
                            page = bm["page"]
                prev_text = ""
                next_text = ""
                chunk_idx = None
                for i, c in enumerate(run_chunks):
                    if c.id == hit.chunk.id:
                        chunk_idx = i
                        if i > 0:
                            prev_text = run_chunks[i - 1].text
                        if i + 1 < len(run_chunks):
                            next_text = run_chunks[i + 1].text
                        break
                item["heading"] = heading
                item["page"] = page
                item["chunk_index"] = chunk_idx
                item["entities"] = ec_map.get(hit.chunk.id, [])[:8]
                item["adjacent"] = {"prev_text": prev_text, "next_text": next_text}
                document_hits.append(item)

        memory_hits: list[dict[str, Any]] = []
        if scope in ("all", "memory") and not document_id:
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

        # Surface clean matching entities & explicit relations
        q_terms = {t.lower() for t in query.split() if len(t) >= 2}
        nodes_map, edges = self._aggregate_graph_data(source=scope, include_cooccurrence=False)
        matched_entities = []
        for nid, node in nodes_map.items():
            if is_single_syllable_vi_noise(node["name"]):
                continue
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
                        "predicate_label": edge.get("predicate_label", edge["predicate"]),
                        "object": nodes_map.get(edge["target"], {}).get("name", edge["target"]),
                        "object_id": edge["target"],
                        "weight": edge.get("weight", 1),
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
