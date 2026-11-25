import re

_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----"),
        "[REDACTED:private_key]",
    ),
    (re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}\b"), "[REDACTED:anthropic_key]"),
    (re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"), "[REDACTED:openai_key]"),
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"), "[REDACTED:github_token]"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "[REDACTED:github_pat]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED:aws_access_key]"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "[REDACTED:slack_token]"),
    (re.compile(r"\bmlab_[A-Za-z0-9_-]{16,}\b"), "[REDACTED:memory_lab_key]"),
    (
        re.compile(r"\b(Bearer)\s+[A-Za-z0-9._-]{24,}\b", flags=re.IGNORECASE),
        r"\1 [REDACTED:bearer_token]",
    ),
    (
        re.compile(
            r"\b(password|passwd|secret_key|api_key|access_token)\s*([:=])\s*['\"]?([^\s'\",;]{8,})['\"]?",
            flags=re.IGNORECASE,
        ),
        r"\1\2[REDACTED:secret]",
    ),
]


def scrub_secrets(text: str) -> tuple[str, int]:
    """Deterministically redact common credentials and tokens before persisting memory artifacts.

    Inspired by ai-memory's client-side secret scrubber. Returns (scrubbed_text, redaction_count).
    """
    if not text:
        return text, 0
    total = 0
    result = text
    for pattern, replacement in _SECRET_PATTERNS:
        result, count = pattern.subn(replacement, result)
        total += count
    return result, total
