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


def _is_document_hit_relevant(hit: Evidence, case: BenchmarkCase) -> bool:
    if case.relevant_chunk_ids and hit.chunk.id in case.relevant_chunk_ids:
        return True
    if case.relevant_document_ids and hit.chunk.document_id in case.relevant_document_ids:
        return True
    if case.relevant_block_ids and any(bid in case.relevant_block_ids for bid in hit.chunk.block_ids):
        return True
    if case.relevant_substrings:
        lower = hit.chunk.text.lower()
        if any(sub.lower() in lower for sub in case.relevant_substrings):
            return True
    return False


def _is_memory_hit_relevant(hit: MemorySearchHit, case: BenchmarkCase) -> bool:
    if case.relevant_memory_ids and hit.record.id in case.relevant_memory_ids:
        return True
    if case.relevant_substrings:
        lower = hit.record.content.lower()
        if any(sub.lower() in lower for sub in case.relevant_substrings):
            return True
    return False


def _expected_relevant_count(case: BenchmarkCase, target: str) -> int:
    if target == "documents":
        counts = [
            len(case.relevant_chunk_ids),
            len(case.relevant_document_ids),
            len(case.relevant_block_ids),
            len(case.relevant_substrings),
        ]
    else:
        counts = [len(case.relevant_memory_ids), len(case.relevant_substrings)]
    positive = [c for c in counts if c > 0]
    return min(positive) if positive else 1


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
                flags = [_is_document_hit_relevant(hit, case) for hit in hits]
            else:
                hits = self.memory.search_memory(
                    case.query,
                    strategy=request.memory_strategy,
                    limit=request.k,
                    session_id=request.session_id,
                    temporal_bucket=request.temporal_bucket,
                )
                retrieved_ids = [hit.record.id for hit in hits]
                flags = [_is_memory_hit_relevant(hit, case) for hit in hits]

            latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
            total_rel = _expected_relevant_count(case, request.target)
            matched_ranks = [idx for idx, is_rel in enumerate(flags, start=1) if is_rel]

            case_results.append(
                ExperimentCaseResult(
                    query=case.query,
                    retrieved_ids=retrieved_ids,
                    matched_ranks=matched_ranks,
                    recall_at_k=recall_at_k(flags, total_rel, request.k),
                    mrr=mean_reciprocal_rank(flags, request.k),
                    ndcg_at_k=ndcg_at_k(flags, total_rel, request.k),
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
