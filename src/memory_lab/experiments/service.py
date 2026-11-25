import json
import time
from typing import Any
from uuid import uuid4
from memory_lab.core.models import (
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkDatasetCreate,
    Evidence,
    ExperimentCaseResult,
    ExperimentReport,
    ExperimentRunRequest,
    MemorySearchHit,
)
from memory_lab.documents.pipeline.service import DocumentPipeline
from memory_lab.experiments.metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k
from memory_lab.memory.service import MemoryService
from memory_lab.storage.filesystem.store import FileStore, digest


def _score_document_hits(hits: list[Evidence], case: BenchmarkCase) -> tuple[list[bool], int, int]:
    """Match hits against ground-truth targets, crediting each distinct target at most once.

    Returns (flags, unique_targets_matched, total_targets).
    """
    if case.relevant_chunk_ids:
        remaining = set(case.relevant_chunk_ids)
        total = len(remaining)
        flags: list[bool] = []
        for hit in hits:
            if hit.chunk.id in remaining:
                remaining.remove(hit.chunk.id)
                flags.append(True)
            else:
                flags.append(False)
        return flags, total - len(remaining), total

    if case.relevant_block_ids:
        remaining = set(case.relevant_block_ids)
        total = len(remaining)
        flags = []
        for hit in hits:
            matched = [bid for bid in hit.chunk.block_ids if bid in remaining]
            if matched:
                for bid in matched:
                    remaining.discard(bid)
                flags.append(True)
            else:
                flags.append(False)
        return flags, total - len(remaining), total

    if case.relevant_document_ids:
        remaining = set(case.relevant_document_ids)
        total = len(remaining)
        flags = []
        for hit in hits:
            if hit.chunk.document_id in remaining:
                remaining.remove(hit.chunk.document_id)
                flags.append(True)
            else:
                flags.append(False)
        return flags, total - len(remaining), total

    if case.relevant_substrings:
        remaining = {sub.lower() for sub in case.relevant_substrings if sub}
        total = len(remaining) or 1
        flags = []
        for hit in hits:
            lower = hit.chunk.text.lower()
            matched = [sub for sub in remaining if sub in lower]
            if matched:
                for sub in matched:
                    remaining.discard(sub)
                flags.append(True)
            else:
                flags.append(False)
        return flags, total - len(remaining), total

    return [False] * len(hits), 0, 1


def _score_memory_hits(hits: list[MemorySearchHit], case: BenchmarkCase) -> tuple[list[bool], int, int]:
    """Match memory hits against ground-truth targets, crediting each distinct target at most once."""
    if case.relevant_memory_ids:
        remaining = set(case.relevant_memory_ids)
        total = len(remaining)
        flags: list[bool] = []
        for hit in hits:
            if hit.record.id in remaining:
                remaining.remove(hit.record.id)
                flags.append(True)
            else:
                flags.append(False)
        return flags, total - len(remaining), total

    if case.relevant_substrings:
        remaining = {sub.lower() for sub in case.relevant_substrings if sub}
        total = len(remaining) or 1
        flags = []
        for hit in hits:
            lower = hit.record.content.lower()
            matched = [sub for sub in remaining if sub in lower]
            if matched:
                for sub in matched:
                    remaining.discard(sub)
                flags.append(True)
            else:
                flags.append(False)
        return flags, total - len(remaining), total

    return [False] * len(hits), 0, 1


class ExperimentService:
    """Manages benchmark datasets, experiment runs with config snapshots, and side-by-side comparison."""

    def __init__(self, store: FileStore, pipeline: DocumentPipeline, memory: MemoryService):
        self.store = store
        self.pipeline = pipeline
        self.memory = memory

    def create_dataset(self, payload: BenchmarkDatasetCreate) -> BenchmarkDataset:
        canonical = json.dumps([c.model_dump() for c in payload.cases], sort_keys=True, ensure_ascii=False)
        dataset_hash = digest(canonical.encode())
        dataset = BenchmarkDataset(
            id=uuid4().hex[:12],
            name=payload.name,
            description=payload.description,
            dataset_hash=dataset_hash,
            cases=payload.cases,
        )
        self.store.put(f"experiments/datasets/{dataset.id}.json", dataset.model_dump())
        return dataset

    def get_dataset(self, dataset_id: str) -> BenchmarkDataset:
        return BenchmarkDataset.model_validate(self.store.get(f"experiments/datasets/{dataset_id}.json"))

    def list_datasets(self) -> list[BenchmarkDataset]:
        items = [
            BenchmarkDataset.model_validate(self.store.get(str(path.relative_to(self.store.root))))
            for path in self.store.root.glob("experiments/datasets/*.json")
        ]
        items.sort(key=lambda item: (item.created_at, item.id), reverse=True)
        return items

    def run_experiment(self, request: ExperimentRunRequest) -> ExperimentReport:
        dataset = self.get_dataset(request.dataset_id)
        case_results: list[ExperimentCaseResult] = []

        for case in dataset.cases:
            started = time.perf_counter()
            if request.target == "documents":
                hits = self.pipeline.search(
                    case.query,
                    request.k,
                    document_id=request.document_id,
                    version=request.version,
                    pipeline_run_id=request.pipeline_run_id,
                    mode=request.search_mode,
                )
                retrieved_ids = [hit.chunk.id for hit in hits]
                flags, matched_targets, total_rel = _score_document_hits(hits, case)
            else:
                hits = self.memory.search_memory(
                    case.query,
                    strategy=request.memory_strategy,
                    limit=request.k,
                    session_id=request.session_id,
                    temporal_bucket=request.temporal_bucket,
                )
                retrieved_ids = [hit.record.id for hit in hits]
                flags, matched_targets, total_rel = _score_memory_hits(hits, case)

            latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
            matched_ranks = [idx for idx, is_rel in enumerate(flags, start=1) if is_rel]
            recall_val = round(min(1.0, matched_targets / max(total_rel, 1)), 6)
            # When a single hit covers multiple distinct targets, cap ideal hits to observed/needed hits
            effective_rel = max(len(matched_ranks), total_rel - max(0, matched_targets - len(matched_ranks)))
            ndcg_val = ndcg_at_k(flags, effective_rel, request.k)

            case_results.append(
                ExperimentCaseResult(
                    query=case.query,
                    retrieved_ids=retrieved_ids,
                    matched_ranks=matched_ranks,
                    recall_at_k=recall_val,
                    mrr=mean_reciprocal_rank(flags, request.k),
                    ndcg_at_k=ndcg_val,
                    latency_ms=latency_ms,
                )
            )

        n = len(case_results) or 1
        metrics = {
            "recall_at_k": round(sum(r.recall_at_k for r in case_results) / n, 6),
            "mrr": round(sum(r.mrr for r in case_results) / n, 6),
            "ndcg_at_k": round(sum(r.ndcg_at_k for r in case_results) / n, 6),
            "avg_latency_ms": round(sum(r.latency_ms for r in case_results) / n, 3),
            "case_count": float(len(case_results)),
        }

        config_snapshot = {
            **request.model_dump(),
            "embedding_namespace": self.pipeline.namespace,
            "embedding_identity": self.pipeline.embedding_identity,
            "pipeline_revision": self.pipeline.revision,
            "resolved_memory_strategy": request.memory_strategy or self.memory.default_strategy,
        }

        report = ExperimentReport(
            id=uuid4().hex[:12],
            name=request.name,
            dataset_id=dataset.id,
            dataset_hash=dataset.dataset_hash,
            target=request.target,
            config_snapshot=config_snapshot,
            k=request.k,
            metrics=metrics,
            case_results=case_results,
        )
        self.store.put(f"experiments/reports/{report.id}.json", report.model_dump())
        return report

    def get_report(self, report_id: str) -> ExperimentReport:
        return ExperimentReport.model_validate(self.store.get(f"experiments/reports/{report_id}.json"))

    def list_reports(self) -> list[ExperimentReport]:
        items = [
            ExperimentReport.model_validate(self.store.get(str(path.relative_to(self.store.root))))
            for path in self.store.root.glob("experiments/reports/*.json")
        ]
        items.sort(key=lambda item: (item.created_at, item.id), reverse=True)
        return items

    def compare_reports(self, report_ids: list[str]) -> dict[str, Any]:
        if not report_ids:
            raise ValueError("Provide at least one report ID to compare")
        reports = [self.get_report(rid) for rid in report_ids]
        baseline = reports[0]
        rows: list[dict[str, Any]] = []
        for report in reports:
            deltas = {
                metric: round(report.metrics.get(metric, 0.0) - baseline.metrics.get(metric, 0.0), 6)
                for metric in ("recall_at_k", "mrr", "ndcg_at_k", "avg_latency_ms")
            }
            rows.append(
                {
                    "id": report.id,
                    "name": report.name,
                    "target": report.target,
                    "dataset_id": report.dataset_id,
                    "dataset_hash": report.dataset_hash,
                    "k": report.k,
                    "config_snapshot": report.config_snapshot,
                    "metrics": report.metrics,
                    "delta_vs_baseline": deltas,
                    "created_at": report.created_at,
                }
            )
        same_dataset_hash = len({r.dataset_hash for r in reports}) == 1
        return {
            "baseline_id": baseline.id,
            "same_dataset_hash": same_dataset_hash,
            "reports": rows,
        }
