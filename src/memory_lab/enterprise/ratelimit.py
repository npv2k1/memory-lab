from collections import defaultdict, deque
import threading
import time
from typing import Any


class RateLimiter:
    """Per-tenant fixed/sliding window rate limiter backed by Redis when configured, else in-memory."""

    def __init__(self, *, redis_url: str = "", default_rpm: int = 0):
        self.default_rpm = default_rpm
        self._lock = threading.Lock()
        self._buckets: dict[str, deque[float]] = defaultdict(deque)
        self._redis: Any = None
        if redis_url.strip():
            try:
                import redis  # type: ignore[import-not-found]
                self._redis = redis.Redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
            except Exception:
                self._redis = None

    def check(self, tenant_id: str, limit_rpm: int | None = None) -> tuple[bool, int, int]:
        """Returns (allowed, limit, remaining)."""
        limit = self.default_rpm if limit_rpm is None else limit_rpm
        if limit <= 0:
            return True, 0, 0

        if self._redis is not None:
            try:
                window = int(time.time() // 60)
                key = f"mlab:rl:{tenant_id}:{window}"
                pipe = self._redis.pipeline()
                pipe.incr(key)
                pipe.expire(key, 70)
                count = int(pipe.execute()[0])
                remaining = max(0, limit - count)
                return count <= limit, limit, remaining
            except Exception:
                pass

        now = time.time()
        cutoff = now - 60.0
        with self._lock:
            q = self._buckets[tenant_id]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= limit:
                return False, limit, 0
            q.append(now)
            return True, limit, max(0, limit - len(q))
