from collections import defaultdict
import threading
import time


class MetricsRegistry:
    """Lightweight thread-safe Prometheus / OpenMetrics exposition registry."""

    def __init__(self):
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._requests: dict[tuple[str, str, str], int] = defaultdict(int)
        self._duration_sum: dict[tuple[str, str], float] = defaultdict(float)
        self._duration_count: dict[tuple[str, str], int] = defaultdict(int)
        self._rate_limited: dict[str, int] = defaultdict(int)

    def observe_request(self, method: str, path: str, status_code: int, duration_seconds: float) -> None:
        normalized = self._normalize_path(path)
        with self._lock:
            self._requests[(method, normalized, str(status_code))] += 1
            self._duration_sum[(method, normalized)] += duration_seconds
            self._duration_count[(method, normalized)] += 1

    def observe_rate_limit(self, tenant_id: str) -> None:
        with self._lock:
            self._rate_limited[tenant_id] += 1

    @staticmethod
    def _normalize_path(path: str) -> str:
        parts = path.strip("/").split("/")
        if not parts or parts == [""]:
            return "/"
        out = []
        for seg in parts:
            if len(seg) >= 12 and any(c.isdigit() for c in seg):
                out.append(":id")
            else:
                out.append(seg)
        return "/" + "/".join(out)

    def render_prometheus(self, *, extra_gauges: dict[str, float] | None = None) -> str:
        lines: list[str] = [
            "# HELP memory_lab_uptime_seconds Process uptime in seconds",
            "# TYPE memory_lab_uptime_seconds gauge",
            f"memory_lab_uptime_seconds {time.time() - self._started_at:.3f}",
            "# HELP memory_lab_http_requests_total Total HTTP requests handled",
            "# TYPE memory_lab_http_requests_total counter",
        ]
        with self._lock:
            for (method, path, status), count in sorted(self._requests.items()):
                lines.append(
                    f'memory_lab_http_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}'
                )
            lines.append("# HELP memory_lab_http_request_duration_seconds_sum Total HTTP request duration")
            lines.append("# TYPE memory_lab_http_request_duration_seconds_sum counter")
            for (method, path), total in sorted(self._duration_sum.items()):
                lines.append(
                    f'memory_lab_http_request_duration_seconds_sum{{method="{method}",path="{path}"}} {total:.6f}'
                )
            lines.append("# HELP memory_lab_http_request_duration_seconds_count Count of observed HTTP requests")
            lines.append("# TYPE memory_lab_http_request_duration_seconds_count counter")
            for (method, path), count in sorted(self._duration_count.items()):
                lines.append(
                    f'memory_lab_http_request_duration_seconds_count{{method="{method}",path="{path}"}} {count}'
                )
            lines.append("# HELP memory_lab_rate_limited_total Total rate-limited requests per tenant")
            lines.append("# TYPE memory_lab_rate_limited_total counter")
            for tenant_id, count in sorted(self._rate_limited.items()):
                lines.append(f'memory_lab_rate_limited_total{{tenant="{tenant_id}"}} {count}')

        if extra_gauges:
            for name, value in sorted(extra_gauges.items()):
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value}")

        return "\n".join(lines) + "\n"
