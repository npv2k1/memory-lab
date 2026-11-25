from collections import Counter
from datetime import datetime, timezone
import math
from memory_lab.core.models import MemoryRecord, TemporalAnchor, TemporalSummaryNode
from memory_lab.storage.filesystem.store import FileStore


def parse_timestamp(ts: str) -> datetime:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return datetime.now(timezone.utc)


def extract_buckets(record: MemoryRecord) -> dict[str, str]:
    dt = parse_timestamp(record.timestamp)
    year = f"{dt.year:04d}"
    month = f"{dt.year:04d}-{dt.month:02d}"
    day = f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"
    session = f"{day}/session:{record.session_id}"
    return {"year": year, "month": month, "day": day, "session": session}


def record_matches_bucket(record: MemoryRecord, bucket: str) -> bool:
    if not bucket:
        return True
    buckets = extract_buckets(record)
    if bucket in buckets.values():
        return True
    if any(value.startswith(bucket) for value in buckets.values()):
        return True
    return any(anchor.normalized.startswith(bucket) for anchor in record.temporal)


class TemporalHierarchy:
    """Maintains a year -> month -> day -> session hierarchy over active MemoryRecords."""

    def __init__(self, store: FileStore):
        self.store = store
        self._key = "memory/temporal/hierarchy.json"

    def rebuild(self, records: list[MemoryRecord]) -> dict[str, list[dict]]:
        active = [r for r in records if not r.deleted]
        by_level: dict[str, dict[str, list[MemoryRecord]]] = {
            "year": {},
            "month": {},
            "day": {},
            "session": {},
        }
        children_map: dict[str, set[str]] = {}

        for record in active:
            b = extract_buckets(record)
            for level in ("year", "month", "day", "session"):
                by_level[level].setdefault(b[level], []).append(record)
            children_map.setdefault(b["year"], set()).add(b["month"])
            children_map.setdefault(b["month"], set()).add(b["day"])
            children_map.setdefault(b["day"], set()).add(b["session"])

        hierarchy: dict[str, list[dict]] = {}
        for level in ("year", "month", "day", "session"):
            nodes: list[dict] = []
            for bucket, bucket_records in sorted(by_level[level].items(), reverse=True):
                entity_counts = Counter(e.name for r in bucket_records for e in r.entities)
                keyword_counts = Counter(k for r in bucket_records for k in r.keywords)
                node = TemporalSummaryNode(
                    bucket=bucket,
                    granularity=level,  # type: ignore[arg-type]
                    record_ids=[r.id for r in bucket_records],
                    entities=[name for name, _ in entity_counts.most_common(8)],
                    keywords=[kw for kw, _ in keyword_counts.most_common(10)],
                    record_count=len(bucket_records),
                    children=sorted(children_map.get(bucket, set()), reverse=True),
                )
                nodes.append(node.model_dump())
            hierarchy[level] = nodes

        self.store.put(self._key, hierarchy)
        return hierarchy

    def load(self) -> dict[str, list[dict]]:
        if not self.store.path(self._key).exists():
            return {"year": [], "month": [], "day": [], "session": []}
        return self.store.get(self._key)

    def score_records(
        self,
        records: list[MemoryRecord],
        *,
        query_temporal: list[TemporalAnchor],
        temporal_bucket: str | None = None,
        now: datetime | None = None,
    ) -> list[tuple[str, float]]:
        ref_now = now or datetime.now(timezone.utc)
        query_norms = {a.normalized for a in query_temporal}
        scored: list[tuple[str, float]] = []

        for record in records:
            if record.deleted:
                continue
            if temporal_bucket and not record_matches_bucket(record, temporal_bucket):
                continue

            age_days = max(0.0, (ref_now - parse_timestamp(record.timestamp)).total_seconds() / 86400.0)
            recency = math.exp(-age_days / 30.0)

            anchor_bonus = 0.0
            if query_norms:
                record_norms = {a.normalized for a in record.temporal} | set(extract_buckets(record).values())
                for qn in query_norms:
                    if any(rn.startswith(qn) or qn.startswith(rn) for rn in record_norms):
                        anchor_bonus += 1.0

            score = round(anchor_bonus + 0.25 * recency, 6)
            if anchor_bonus > 0 or temporal_bucket:
                scored.append((record.id, score))

        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored
