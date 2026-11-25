from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any, Literal
from memory_lab.core.models import MemoryRecord
from memory_lab.documents.enrichment.extractor import STOPWORDS, tokenize

FeedbackSignal = Literal["helpful", "not_helpful", "stale", "wrong"]

SALIENCE_MIN = 0.25
SALIENCE_MAX = 2.0
SALIENCE_STEP = 0.25

FUNCTIONAL_PREDICATES = frozenset({
    "reports_to",
    "works_at",
    "deployed_on",
    "manages",
})


@dataclass(frozen=True)
class DecayParams:
    """Zero-LLM memory aging parameters modeled after akitaonrails/ai-memory."""

    lambda_decay: float = 0.02  # ~35-day half-life for episodic decay
    sigma_access: float = 0.6
    mu_access: float = 0.04
    cold_threshold: float = 0.20
    access_bump_cooldown_secs: int = 60


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def is_expired(expires_at: str | None, *, now: datetime | None = None) -> bool:
    exp = _parse_iso(expires_at)
    if exp is None:
        return False
    ref = now or datetime.now(timezone.utc)
    return exp <= ref


def retention_score(
    record: MemoryRecord,
    *,
    params: DecayParams | None = None,
    now: datetime | None = None,
) -> float:
    """Compute pure retention score:
    retention = salience * exp(-lambda * age_days)
              + sigma * ln(1 + access_count) * exp(-mu * days_since_access)
    """
    cfg = params or DecayParams()
    ref = now or datetime.now(timezone.utc)
    created = _parse_iso(record.timestamp) or ref
    age_days = max(0.0, (ref - created).total_seconds() / 86400.0)

    base = float(record.salience) * math.exp(-cfg.lambda_decay * age_days)

    boost = 0.0
    if record.access_count > 0:
        last_acc = _parse_iso(record.last_accessed_at) or created
        days_since_access = max(0.0, (ref - last_acc).total_seconds() / 86400.0)
        boost = cfg.sigma_access * math.log(1.0 + record.access_count) * math.exp(-cfg.mu_access * days_since_access)

    return round(base + boost, 6)


def salience_after_feedback(current: float, signal: FeedbackSignal) -> float:
    """Adjust memory salience monotonically on helpful/not_helpful, or drop to floor on stale/wrong."""
    if signal == "helpful":
        return round(min(SALIENCE_MAX, current + SALIENCE_STEP), 4)
    if signal == "not_helpful":
        return round(max(SALIENCE_MIN, current - SALIENCE_STEP), 4)
    if signal in {"stale", "wrong"}:
        return SALIENCE_MIN
    raise ValueError(f"Unsupported feedback signal: {signal}")


def evaluate_eviction(
    record: MemoryRecord,
    *,
    params: DecayParams | None = None,
    now: datetime | None = None,
) -> tuple[bool, str, float]:
    """Determine if a record is eligible for reversible archival during a forget sweep.

    Returns (should_archive, reason, retention).
    """
    cfg = params or DecayParams()
    ref = now or datetime.now(timezone.utc)
    score = retention_score(record, params=cfg, now=ref)

    if is_expired(record.expires_at, now=ref):
        return True, "ttl_expired", score
    if record.pinned:
        return False, "pinned_exempt", score
    if record.tier in {"semantic", "procedural"}:
        return False, f"tier_{record.tier}_exempt", score
    if score < cfg.cold_threshold:
        return True, "cold_decay", score
    return False, "active", score


def _content_tokens(text: str) -> set[str]:
    return {
        tok
        for tok in tokenize(text, include_folded=False)
        if tok not in STOPWORDS and len(tok) >= 2
    }


def lint_memories(
    records: list[MemoryRecord],
    *,
    params: DecayParams | None = None,
    duplicate_threshold: float = 0.72,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Zero-LLM memory hygiene audit: detects near-duplicates, graph contradictions, stale/cold memories."""
    cfg = params or DecayParams()
    ref = now or datetime.now(timezone.utc)

    active = [r for r in records if not r.deleted and not r.archived]
    token_sets = {r.id: _content_tokens(r.content) for r in active}

    # 1. Near-duplicate detection via Jaccard similarity
    near_duplicates: list[dict[str, Any]] = []
    for i in range(len(active)):
        r1 = active[i]
        t1 = token_sets[r1.id]
        if len(t1) < 3:
            continue
        for j in range(i + 1, len(active)):
            r2 = active[j]
            if r1.project != r2.project:
                continue
            t2 = token_sets[r2.id]
            if len(t2) < 3:
                continue
            union = len(t1 | t2)
            if union == 0:
                continue
            jaccard = len(t1 & t2) / union
            if jaccard >= duplicate_threshold:
                near_duplicates.append(
                    {
                        "record_a": r1.id,
                        "record_b": r2.id,
                        "similarity": round(jaccard, 4),
                        "project": r1.project,
                        "preview_a": r1.summary or r1.content[:120],
                        "preview_b": r2.summary or r2.content[:120],
                    }
                )

    # 2. Contradiction detection over functional SPO relations (same project + subject + predicate -> different objects)
    by_sp: dict[tuple[str, str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for rec in active:
        for rel in rec.relations:
            if rel.predicate in FUNCTIONAL_PREDICATES:
                by_sp[(rec.project, rel.subject, rel.predicate)].append((rel.object, rec.id, rec.summary or rec.content[:120]))

    contradictions: list[dict[str, Any]] = []
    for (proj, subj, pred), entries in by_sp.items():
        distinct_objects: dict[str, tuple[str, str]] = {}
        for obj, rid, preview in entries:
            distinct_objects.setdefault(obj, (rid, preview))
        if len(distinct_objects) > 1:
            contradictions.append(
                {
                    "project": proj,
                    "subject": subj,
                    "predicate": pred,
                    "conflicting_values": [
                        {"object": obj, "record_id": rid, "preview": prev}
                        for obj, (rid, prev) in sorted(distinct_objects.items())
                    ],
                }
            )

    # 3. Stale, flagged, expired, or cold-decay candidates
    cold_or_stale: list[dict[str, Any]] = []
    for rec in active:
        should_evict, reason, score = evaluate_eviction(rec, params=cfg, now=ref)
        flagged_signals = [
            fb.get("signal")
            for fb in rec.feedback_history
            if fb.get("signal") in {"stale", "wrong"}
        ]
        if should_evict or flagged_signals:
            cold_or_stale.append(
                {
                    "record_id": rec.id,
                    "project": rec.project,
                    "tier": rec.tier,
                    "salience": rec.salience,
                    "retention_score": score,
                    "reason": reason if should_evict else f"feedback:{flagged_signals[-1]}",
                    "preview": rec.summary or rec.content[:120],
                }
            )

    return {
        "checked_records": len(active),
        "near_duplicates": near_duplicates,
        "contradictions": contradictions,
        "cold_or_stale": cold_or_stale,
        "issue_count": len(near_duplicates) + len(contradictions) + len(cold_or_stale),
    }
