from collections import Counter
from datetime import datetime, timedelta, timezone
import re
import unicodedata
from memory_lab.core.models import (
    Chunk,
    EnrichedChunk,
    EnrichmentArtifact,
    Entity,
    Relation,
    TemporalAnchor,
)

STOPWORDS = frozenset({
    # English
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "here",
    "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just",
    "may", "me", "more", "most", "my", "no", "not", "of", "on", "or", "our", "out",
    "over", "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "up", "us", "very", "was", "we", "were", "what", "when", "where", "which",
    "who", "whom", "whose", "why", "will", "with", "would", "you", "your", "about",
    "also", "into", "only", "still", "well", "such", "then", "than", "both", "each",
    # Vietnamese (accented + folded)
    "va", "cua", "la", "cac", "co", "duoc", "cho", "trong", "mot", "nhung", "nay",
    "khi", "voi", "tu", "den", "tai", "ve", "nhu", "khong", "da", "se", "dang", "theo",
    "qua", "sau", "truoc", "de", "neu", "thi", "hay", "hoac", "boi", "vi", "nen",
    "và", "của", "là", "các", "có", "được", "cho", "trong", "một", "những", "này",
    "với", "từ", "đến", "tại", "về", "như", "không", "đã", "sẽ", "đang", "để", "nếu",
    "thì", "hoặc", "bởi", "vì", "nên", "trên", "dưới", "giữa", "cùng", "cũng", "vẫn",
    "tôi", "toi", "mình", "minh", "bạn", "ban", "anh", "chị", "chi", "em", "chúng",
    "chung", "họ", "ho", "nó", "rất", "rat", "lắm", "lam", "quá", "nữa", "nua",
    "đều", "deu", "chỉ", "mới", "moi", "nào", "nao", "gì", "gi", "ai", "đâu", "dau",
    "sao", "thế", "the", "vậy", "vay", "nhé", "nhe", "nhỉ", "nhi", "hãy", "đừng",
    "dung", "phải", "phai", "cần", "can", "muốn", "muon", "lại", "lai", "rồi", "roi",
    "chưa", "chua", "từng", "tung", "mỗi", "mọi", "nhiều", "nhieu", "hơn", "hon",
    "nhất", "nhat", "bằng", "bang", "đó", "do", "đây", "day", "kia", "ấy", "ay",
    "nhưng", "nhung", "tuy", "ngoài", "ngoai", "vào", "vao", "lên", "len", "xuống",
    "xuong", "ra", "đi", "di", "về", "ở", "tất", "cả", "mà", "ma", "hay",
})

# Capitalized words that commonly start English/Vietnamese sentences and are NOT named entities
SENTENCE_START_NON_ENTITIES = frozenset({
    "the", "this", "that", "these", "those", "when", "where", "what", "which", "who",
    "whom", "whose", "why", "how", "however", "therefore", "moreover", "furthermore",
    "meanwhile", "additionally", "finally", "first", "second", "third", "last", "next",
    "then", "there", "here", "after", "before", "while", "during", "since", "until",
    "although", "though", "because", "unless", "whether", "also", "still", "just",
    "only", "even", "both", "each", "every", "all", "some", "many", "most", "few",
    "several", "other", "another", "such", "new", "old", "good", "bad", "high", "low",
    "full", "open", "free", "fast", "slow", "best", "main", "core", "key", "top",
    "note", "tip", "warning", "caution", "important", "example", "summary", "overview",
    "section", "chapter", "part", "step", "steps", "stage", "stages", "phase", "table",
    "figure", "list", "item", "data", "file", "files", "code", "test", "tests", "run",
    "runs", "use", "using", "used", "add", "set", "get", "put", "post", "delete",
    "create", "update", "remove", "check", "start", "stop", "build", "deploy", "please",
    "see", "read", "write", "store", "search", "query", "memory", "system", "document",
    "documents", "chunk", "chunks", "entity", "entities", "graph", "vector", "hybrid",
    "sparse", "dense", "local", "remote", "default", "custom", "current", "previous",
    "enable", "disable", "configure", "install", "support", "supports", "provides",
    "returns", "allows", "requires", "includes", "contains", "ensure", "make", "keep",
    "we", "they", "he", "she", "it", "you", "our", "my", "your", "their", "his", "her",
    "yes", "no", "not", "never", "always", "often", "sometimes", "usually", "currently",
    "today", "yesterday", "tomorrow", "now", "soon", "later", "earlier", "recently",
    # Vietnamese sentence starters
    "hệ", "hệ thống", "người", "người dùng", "tôi", "mình", "bạn", "chúng", "chúng tôi",
    "chúng ta", "họ", "anh", "chị", "em", "khi", "nếu", "sau", "trước", "trong", "trên",
    "dưới", "từ", "đến", "với", "cho", "của", "các", "những", "một", "này", "đó", "đây",
    "là", "có", "được", "không", "đã", "sẽ", "đang", "vì", "nên", "nhưng", "tuy", "ngoài",
    "theo", "về", "tại", "lưu", "lưu ý", "chú", "chú ý", "xem", "bước", "phần", "mục",
    "câu", "câu hỏi", "tài", "tài liệu", "dữ", "dữ liệu", "kết", "kết quả", "thông",
    "thông tin", "hướng", "hướng dẫn", "cách", "việc", "quản", "quản lý", "dự", "dự án",
    "công", "hỗ", "hỗ trợ", "tìm", "tìm kiếm", "chạy", "cài", "cài đặt", "cấu", "cấu hình",
    "hôm", "hôm nay", "hôm qua", "ngày", "ngày mai", "tuần", "tháng", "năm",
})

KNOWN_TECH: dict[str, tuple[str, str]] = {
    "qdrant": ("Qdrant", "technology"),
    "neo4j": ("Neo4j", "technology"),
    "redis": ("Redis", "technology"),
    "fastapi": ("FastAPI", "technology"),
    "astro": ("Astro", "technology"),
    "docker": ("Docker", "technology"),
    "docling": ("Docling", "technology"),
    "pytorch": ("PyTorch", "technology"),
    "playwright": ("Playwright", "technology"),
    "python": ("Python", "technology"),
    "typescript": ("TypeScript", "technology"),
    "javascript": ("JavaScript", "technology"),
    "bun": ("Bun", "technology"),
    "nginx": ("Nginx", "technology"),
    "caddy": ("Caddy", "technology"),
    "bm25": ("BM25", "technology"),
    "mcp": ("MCP", "technology"),
    "rag": ("RAG", "concept"),
    "llm": ("LLM", "concept"),
    "openai": ("OpenAI", "organization"),
    "anthropic": ("Anthropic", "organization"),
    "google": ("Google", "organization"),
    "microsoft": ("Microsoft", "organization"),
    "apple": ("Apple", "organization"),
    "github": ("GitHub", "organization"),
    "memory lab": ("Memory Lab", "concept"),
    "vectormemory": ("VectorMemory", "technology"),
    "graphmemory": ("GraphMemory", "technology"),
    "hybridmemory": ("HybridMemory", "technology"),
    "zerotokenmemory": ("ZeroTokenMemory", "technology"),
    "ubuntu": ("Ubuntu", "technology"),
    "linux": ("Linux", "technology"),
    "postgresql": ("PostgreSQL", "technology"),
    "postgres": ("PostgreSQL", "technology"),
    "mysql": ("MySQL", "technology"),
    "sqlite": ("SQLite", "technology"),
    "mongodb": ("MongoDB", "technology"),
    "ollama": ("Ollama", "technology"),
    "bge-m3": ("bge-m3", "technology"),
    "nomic-embed-text": ("nomic-embed-text", "technology"),
    "multilingual-e5": ("multilingual-e5", "technology"),
    "cloud run": ("Cloud Run", "technology"),
    "google cloud": ("Google Cloud", "organization"),
    "kubernetes": ("Kubernetes", "technology"),
    "minio": ("MinIO", "technology"),
    "prometheus": ("Prometheus", "technology"),
    "grafana": ("Grafana", "technology"),
    "vdaily": ("vdaily", "concept"),
    "mem0": ("Mem0", "technology"),
    "zep": ("Zep", "technology"),
    "letta": ("Letta", "technology"),
    "cognee": ("Cognee", "technology"),
    "langchain": ("LangChain", "technology"),
    "llamaindex": ("LlamaIndex", "technology"),
    "huggingface": ("HuggingFace", "organization"),
    "vllm": ("vLLM", "technology"),
    "claude": ("Claude", "technology"),
    "gemini": ("Gemini", "technology"),
    "chatgpt": ("ChatGPT", "technology"),
    "react": ("React", "technology"),
    "nextjs": ("Next.js", "technology"),
    "vite": ("Vite", "technology"),
    "celery": ("Celery", "technology"),
    "kafka": ("Kafka", "technology"),
}

TECH_ACRONYMS = frozenset({
    "HTTP", "HTTPS", "JSON", "YAML", "TOML", "UUID", "UTF", "URL", "URI",
    "GET", "POST", "PUT", "PATCH", "DELETE", "API", "REST", "RPC", "GRPC",
    "SDK", "CLI", "GUI", "UI", "UX", "SQL", "NOSQL", "CSV", "TSV", "PDF",
    "DOCX", "HTML", "CSS", "XML", "SSE", "WS", "TLS", "SSL", "JWT", "RBAC",
    "CORS", "CI", "CD", "CPU", "GPU", "RAM", "VRAM", "SSD", "OS", "VM",
    "DNS", "IP", "TCP", "UDP", "SSH", "S3", "MCP", "OCR", "BM25", "RRF",
    "UMAP", "PCA", "SVD", "KNN", "HNSW", "ANN", "LRU", "TTL", "RPM", "ENV",
    "ORM", "CRUD", "ACID", "SSO", "OIDC", "SAML", "AWS", "GCP", "K8S", "CDN",
    "VPC", "IAM", "KMS", "WAF", "E2E", "QA", "ML", "AI", "DL", "CV", "NLP",
})

CONCEPT_ACRONYMS = frozenset({
    "RAG", "LLM", "NER", "MRR", "NDCG", "SLO", "SLA", "KPI", "OKR", "ROI",
    "MVP", "POC", "PRD", "RFC", "ADR", "PM", "CTO", "CEO", "VP", "HR",
})

ORG_SUFFIXES = (
    " corp", " inc", " ltd", " llc", " labs", " lab", " technologies", " tech",
    " systems", " solutions", " group", " studio", " foundation", " bank",
    " university", " company", " co", " ventures", " capital", " ai",
)

MONTH_MAP = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

# Unicode-aware entity span (1 to 4 capitalized/acronym/hyphenated tokens)
_ENT = r"[A-ZÀ-Ỹ][\wÀ-ỹ.-]*(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ.-]*){0,3}"
_ENT_FLEX = r"[A-ZÀ-Ỹ][\wÀ-ỹ.-]*(?:\s+(?:of|for|the)?\s*[A-ZÀ-Ỹ][\wÀ-ỹ.-]*){0,3}"

RELATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Organizational & management relations (EN + VI)
    (re.compile(rf"\b({_ENT})\s+reports\s+to\s+({_ENT})\b"), "reports_to"),
    (re.compile(rf"\b({_ENT})\s+(?:báo cáo cho|trực thuộc)\s+({_ENT})\b"), "reports_to"),
    (
        re.compile(
            rf"\b({_ENT})\s+is\s+(?:the\s+|a\s+)?(?:project\s+|engineering\s+|product\s+|team\s+|tech\s+|technical\s+)?"
            rf"(?:manager|lead|leader|director|head|owner|architect)\s+(?:of|for)\s+({_ENT_FLEX})\b"
        ),
        "manages",
    ),
    (re.compile(rf"\b({_ENT})\s+(?:manages|leads|directs|oversees|owns)\s+({_ENT_FLEX})\b"), "manages"),
    (
        re.compile(
            rf"\b({_ENT})\s+(?:là\s+)?(?:quản lý|trưởng nhóm|phụ trách|đứng đầu|chủ trì)(?:\s+dự án)?(?:\s+của)?\s+({_ENT_FLEX})\b"
        ),
        "manages",
    ),
    (re.compile(rf"\b({_ENT})\s+works\s+(?:at|for)\s+({_ENT_FLEX})\b"), "works_at"),
    (re.compile(rf"\b({_ENT})\s+làm việc\s+(?:tại|cho|ở)\s+({_ENT_FLEX})\b"), "works_at"),
    (re.compile(rf"\b({_ENT})\s+works\s+on\s+({_ENT_FLEX})\b"), "works_on"),
    (re.compile(rf"\b({_ENT})\s+(?:tham gia|phát triển)\s+({_ENT_FLEX})\b"), "works_on"),
    # Deployment & preferences
    (re.compile(rf"\b({_ENT})\s+(?:is\s+)?(?:deployed|hosted|runs|running)\s+on\s+({_ENT_FLEX})\b"), "deployed_on"),
    (re.compile(rf"\b({_ENT})\s+(?:được\s+)?(?:triển khai|chạy)\s+trên\s+({_ENT_FLEX})\b"), "deployed_on"),
    (re.compile(rf"\b({_ENT})\s+(?:prefers|likes|loves)\s+({_ENT_FLEX})\b"), "prefers"),
    (re.compile(rf"\b({_ENT})\s+(?:rất\s+)?thích\s+({_ENT_FLEX})\b"), "prefers"),
    # Technical architecture relations (EN + VI)
    (re.compile(rf"\b({_ENT})\s+(?:uses|using|use)\s+({_ENT_FLEX})\b"), "uses"),
    (re.compile(rf"\b({_ENT})\s+(?:depends on|relies on)\s+({_ENT_FLEX})\b"), "depends_on"),
    (re.compile(rf"\b({_ENT})\s+(?:connects to|integrates with)\s+({_ENT_FLEX})\b"), "integrates_with"),
    (re.compile(rf"\b({_ENT})\s+(?:stores|indexes|persists)\s+({_ENT_FLEX})\b"), "stores"),
    (re.compile(rf"\b({_ENT})\s+(?:created|built|authored|maintains|developed)\s+({_ENT_FLEX})\b"), "created"),
    (re.compile(rf"\b({_ENT})\s+(?:is a|is an|extends|implements)\s+({_ENT_FLEX})\b"), "is_a"),
    (re.compile(rf"\b({_ENT})\s+(?:sử dụng|dùng)\s+({_ENT_FLEX})\b"), "uses"),
    (re.compile(rf"\b({_ENT})\s+(?:lưu trữ|lưu)\s+({_ENT_FLEX})\b"), "stores"),
    (re.compile(rf"\b({_ENT})\s+(?:kết nối với|kết nối)\s+({_ENT_FLEX})\b"), "integrates_with"),
    (re.compile(rf"\b({_ENT})\s+(?:phụ thuộc vào|phụ thuộc)\s+({_ENT_FLEX})\b"), "depends_on"),
]

PERSON_PREDICATES = frozenset({"reports_to", "manages", "works_at", "works_on", "prefers"})


def fold_ascii(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).replace("đ", "d").replace("Đ", "D")


def normalize_name(name: str) -> str:
    cleaned = re.sub(r"['’]s\b", "", name.strip())
    return re.sub(r"\s+", " ", cleaned).strip(" ,:;()[]{}\"'`").lower()


def tokenize(text: str, *, include_folded: bool = True) -> list[str]:
    cleaned = re.sub(r"['’]s\b", "", text)
    raw_tokens = re.findall(r"[^\W_]+", cleaned.lower(), flags=re.UNICODE)
    tokens: list[str] = []
    for token in raw_tokens:
        if len(token) < 2:
            continue
        tokens.append(token)
        if include_folded:
            folded = fold_ascii(token)
            if folded and folded != token:
                tokens.append(folded)
    return tokens


def _parse_reference_date(reference_timestamp: str | None) -> datetime:
    if not reference_timestamp:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(reference_timestamp.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)


def _strip_possessive(word: str) -> str:
    return re.sub(r"['’]s$", "", word)


def _is_standalone_occurrence(text_lower: str, short_norm: str, longer_norms: list[str]) -> bool:
    """Return True if `short_norm` appears at least once outside all `longer_norms` spans."""
    masked = text_lower
    for long_norm in longer_norms:
        masked = re.sub(rf"(?<!\w){re.escape(long_norm)}(?!\w)", " ", masked)
    return bool(re.search(rf"(?<!\w){re.escape(short_norm)}(?!\w)", masked))


def _classify_phrase(name: str, norm: str) -> str:
    if norm in KNOWN_TECH:
        return KNOWN_TECH[norm][1]
    if any(norm.endswith(suf) for suf in ORG_SUFFIXES) or norm.startswith(("công ty ", "tập đoàn ")):
        return "organization"
    if norm.startswith(("project ", "dự án ")):
        return "concept"
    return "concept"


class NonGenerativeExtractor:
    """Deterministic, zero-LLM extractor for entities, relations, temporal anchors, and keywords."""

    def extract_temporal(self, text: str, reference_timestamp: str | None = None) -> list[TemporalAnchor]:
        anchors: dict[tuple[str, str], TemporalAnchor] = {}
        ref = _parse_reference_date(reference_timestamp)

        # ISO YYYY-MM-DD
        for match in re.finditer(r"\b(19\d{2}|20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b", text):
            raw = match.group(0)
            anchors[(raw, "day")] = TemporalAnchor(raw=raw, normalized=raw, granularity="day")

        # DD/MM/YYYY
        for match in re.finditer(r"\b(0?[1-9]|[12]\d|3[01])/(0?[1-9]|1[0-2])/(19\d{2}|20\d{2})\b", text):
            day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
            norm = f"{year:04d}-{month:02d}-{day:02d}"
            anchors[(norm, "day")] = TemporalAnchor(raw=match.group(0), normalized=norm, granularity="day")

        # YYYY-MM (not followed by -DD)
        for match in re.finditer(r"\b(19\d{2}|20\d{2})-(0[1-9]|1[0-2])\b(?!-\d{2})", text):
            raw = match.group(0)
            anchors[(raw, "month")] = TemporalAnchor(raw=raw, normalized=raw, granularity="month")

        # English month + year
        month_names = "|".join(MONTH_MAP.keys())
        for match in re.finditer(rf"\b({month_names})\s+(19\d{{2}}|20\d{{2}})\b", text, flags=re.IGNORECASE):
            month_num = MONTH_MAP[match.group(1).lower()]
            year = int(match.group(2))
            norm = f"{year:04d}-{month_num:02d}"
            anchors[(norm, "month")] = TemporalAnchor(raw=match.group(0), normalized=norm, granularity="month")

        # Vietnamese tháng M năm YYYY or tháng M/YYYY
        for match in re.finditer(r"\btháng\s+(0?[1-9]|1[0-2])\s*(?:năm\s+|/)\s*(19\d{2}|20\d{2})\b", text, flags=re.IGNORECASE):
            month_num, year = int(match.group(1)), int(match.group(2))
            norm = f"{year:04d}-{month_num:02d}"
            anchors[(norm, "month")] = TemporalAnchor(raw=match.group(0), normalized=norm, granularity="month")

        # Standalone years not already covered by a full date/month
        covered_years = {a.normalized[:4] for a in anchors.values()}
        for match in re.finditer(r"\b(19\d{2}|20\d{2})\b", text):
            year = match.group(1)
            if year not in covered_years:
                anchors[(year, "year")] = TemporalAnchor(raw=year, normalized=year, granularity="year")

        # Relative temporal cues
        relative_offsets = [
            (r"\b(?:today|hôm nay)\b", 0),
            (r"\b(?:yesterday|hôm qua)\b", -1),
            (r"\b(?:tomorrow|ngày mai)\b", 1),
            (r"\b(?:last week|tuần trước)\b", -7),
        ]
        for pattern, delta_days in relative_offsets:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                resolved = (ref + timedelta(days=delta_days)).strftime("%Y-%m-%d")
                anchors[(resolved, "relative")] = TemporalAnchor(
                    raw=match.group(0), normalized=resolved, granularity="relative"
                )

        return sorted(anchors.values(), key=lambda item: (item.normalized, item.granularity))

    def extract_entities(self, text: str) -> list[Entity]:
        found: dict[str, Entity] = {}
        lowered = text.lower()

        # 1. Known technical/domain entities
        for key, (canonical, etype) in KNOWN_TECH.items():
            if re.search(rf"(?<!\w){re.escape(key)}(?!\w)", lowered):
                norm = normalize_name(canonical)
                found[norm] = Entity(name=canonical, normalized_name=norm, type=etype)

        # 2. CamelCase identifiers (e.g., DocumentPipeline, FileStore)
        for match in re.finditer(r"\b[A-Z][a-z]+[A-Z][A-Za-z0-9]*\b", text):
            name = match.group(0)
            norm = normalize_name(name)
            if norm not in found:
                found[norm] = Entity(name=name, normalized_name=norm, type="technology")

        # 3. Acronyms (2-8 uppercase letters/digits)
        for match in re.finditer(r"\b[A-Z][A-Z0-9]{1,7}\b", text):
            name = match.group(0)
            norm = normalize_name(name)
            if norm in found or norm in STOPWORDS or norm in SENTENCE_START_NON_ENTITIES:
                continue
            if name in TECH_ACRONYMS:
                etype = "technology"
            elif name in CONCEPT_ACRONYMS:
                etype = "concept"
            else:
                etype = "organization"
            found[norm] = Entity(name=name, normalized_name=norm, type=etype)

        # 4. Multi-word capitalized proper noun phrases (EN + VI)
        for match in re.finditer(r"\b[A-ZÀ-Ỹ][a-zà-ỹ0-9]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ0-9]+){1,3}\b", text):
            raw_phrase = match.group(0).strip()
            # Trim leading sentence-starter non-entity word if phrase has >=2 words (e.g. "When Alice" -> "Alice")
            parts = raw_phrase.split()
            while (
                len(parts) > 1
                and normalize_name(" ".join(parts)) not in KNOWN_TECH
                and parts[0].lower() in SENTENCE_START_NON_ENTITIES
            ):
                parts.pop(0)
            name = " ".join(parts)
            norm = normalize_name(name)
            if not norm or norm in found or norm in STOPWORDS or norm in SENTENCE_START_NON_ENTITIES:
                continue
            found[norm] = Entity(name=name, normalized_name=norm, type=_classify_phrase(name, norm))  # type: ignore[arg-type]

        # 5. Capitalized single proper nouns (including sentence-initial names like Alice, Bob, Nguyên)
        for sentence in re.split(r"[.!?\n]+", text):
            words = sentence.strip().split()
            for idx, raw_word in enumerate(words):
                clean = _strip_possessive(raw_word.strip(",:;()[]{}\"'`"))
                if not re.fullmatch(r"[A-ZÀ-Ỹ][a-zà-ỹ]{1,}", clean):
                    continue
                norm = normalize_name(clean)
                if (
                    not norm
                    or norm in found
                    or norm in STOPWORDS
                    or norm in MONTH_MAP
                    or norm in SENTENCE_START_NON_ENTITIES
                ):
                    continue
                # Sentence-initial 2-letter capitalized words are too ambiguous unless known
                if idx == 0 and len(clean) < 3:
                    continue
                found[norm] = Entity(name=clean, normalized_name=norm, type="concept")

        # 6. Deduplicate subsumed shorter fragments (e.g., drop "cloud" / "run" when only inside "cloud run")
        all_norms = list(found.keys())
        to_remove: set[str] = set()
        for short_norm in all_norms:
            covering = [
                other for other in all_norms
                if other != short_norm
                and len(other) > len(short_norm)
                and re.search(rf"(?<!\w){re.escape(short_norm)}(?!\w)", other)
            ]
            if covering and not _is_standalone_occurrence(lowered, short_norm, covering):
                to_remove.add(short_norm)

        for norm in to_remove:
            found.pop(norm, None)

        return sorted(found.values(), key=lambda e: e.normalized_name)

    def extract_relations(self, text: str, entities: list[Entity], evidence_id: str = "") -> list[Relation]:
        relations: dict[tuple[str, str, str], Relation] = {}
        entity_by_norm = {e.normalized_name: e for e in entities}

        # Explicit pattern-based relations
        for pattern, predicate in RELATION_PATTERNS:
            for match in pattern.finditer(text):
                raw_subj = match.group(1).strip()
                raw_obj = match.group(2).strip()
                # Strip leading non-entity determiners from captured spans unless full span is a known entity
                subj_parts = raw_subj.split()
                while (
                    len(subj_parts) > 1
                    and normalize_name(" ".join(subj_parts)) not in KNOWN_TECH
                    and normalize_name(" ".join(subj_parts)) not in entity_by_norm
                    and subj_parts[0].lower() in SENTENCE_START_NON_ENTITIES
                ):
                    subj_parts.pop(0)
                obj_parts = raw_obj.split()
                while (
                    len(obj_parts) > 1
                    and normalize_name(" ".join(obj_parts)) not in KNOWN_TECH
                    and normalize_name(" ".join(obj_parts)) not in entity_by_norm
                    and obj_parts[0].lower() in SENTENCE_START_NON_ENTITIES
                ):
                    obj_parts.pop(0)
                subj = normalize_name(" ".join(subj_parts))
                obj = normalize_name(" ".join(obj_parts))
                if (
                    subj
                    and obj
                    and subj != obj
                    and subj not in STOPWORDS
                    and obj not in STOPWORDS
                    and subj not in SENTENCE_START_NON_ENTITIES
                    and obj not in SENTENCE_START_NON_ENTITIES
                ):
                    key = (subj, predicate, obj)
                    relations[key] = Relation(
                        subject=subj, predicate=predicate, object=obj, evidence_id=evidence_id, confidence=0.92
                    )
                    # Upgrade person typing when an entity acts as a person subject/object
                    if predicate in PERSON_PREDICATES:
                        subj_ent = entity_by_norm.get(subj)
                        if subj_ent is not None and subj_ent.type == "concept":
                            subj_ent.type = "person"
                        if predicate == "reports_to":
                            obj_ent = entity_by_norm.get(obj)
                            if obj_ent is not None and obj_ent.type == "concept":
                                obj_ent.type = "person"

        # Sentence-level entity co-occurrence relations (capped to avoid noise)
        entity_norms = [e.normalized_name for e in entities[:12]]
        if len(entity_norms) >= 2:
            for sentence in re.split(r"[.!?\n]+", text):
                lower_sentence = sentence.lower()
                present = [e for e in entity_norms if re.search(rf"(?<!\w){re.escape(e)}(?!\w)", lower_sentence)]
                for i in range(min(len(present), 5)):
                    for j in range(i + 1, min(len(present), 5)):
                        subj, obj = sorted((present[i], present[j]))
                        if not any(
                            {r.subject, r.object} == {subj, obj} for r in relations.values()
                        ):
                            key = (subj, "co_occurs_with", obj)
                            relations[key] = Relation(
                                subject=subj,
                                predicate="co_occurs_with",
                                object=obj,
                                evidence_id=evidence_id,
                                confidence=0.6,
                            )

        return list(relations.values())

    def extract_keywords(self, text: str, limit: int = 12) -> list[str]:
        raw_tokens = re.findall(r"[^\W_]+", re.sub(r"['’]s\b", "", text.lower()), flags=re.UNICODE)
        valid_positions: list[tuple[int, str]] = []
        for idx, tok in enumerate(raw_tokens):
            if len(tok) < 2 or tok.isdigit():
                continue
            if tok in STOPWORDS or fold_ascii(tok) in STOPWORDS:
                continue
            valid_positions.append((idx, tok))

        counts: Counter[str] = Counter()
        # Score adjacent content bigrams (keeps Vietnamese compound words like "cà phê", "quản lý", "dự án")
        bigram_tokens: set[str] = set()
        for i in range(len(valid_positions) - 1):
            pos_a, tok_a = valid_positions[i]
            pos_b, tok_b = valid_positions[i + 1]
            if pos_b == pos_a + 1 and tok_a != tok_b:
                # Prefer bigrams when at least one token has Vietnamese diacritics or both are short syllables
                is_vi = any(ord(ch) > 127 for ch in (tok_a + tok_b))
                if is_vi or (len(tok_a) >= 3 and len(tok_b) >= 3):
                    phrase = f"{tok_a} {tok_b}"
                    counts[phrase] += 2 if is_vi else 1
                    if is_vi:
                        bigram_tokens.add(tok_a)
                        bigram_tokens.add(tok_b)

        for _pos, tok in valid_positions:
            # Slightly down-weight standalone syllables already captured in a Vietnamese compound bigram
            if tok in bigram_tokens and len(tok) <= 4:
                counts[tok] += 1
            else:
                counts[tok] += 2

        ranked: list[str] = []
        seen: set[str] = set()
        for term, _score in counts.most_common(limit * 2):
            if term in seen:
                continue
            seen.add(term)
            ranked.append(term)
            if len(ranked) >= limit:
                break
        return ranked

    def extract_text(
        self, text: str, *, evidence_id: str = "", reference_timestamp: str | None = None
    ) -> tuple[list[Entity], list[Relation], list[TemporalAnchor], list[str]]:
        entities = self.extract_entities(text)
        relations = self.extract_relations(text, entities, evidence_id=evidence_id)
        temporal = self.extract_temporal(text, reference_timestamp=reference_timestamp)
        keywords = self.extract_keywords(text)
        return entities, relations, temporal, keywords

    def extract_chunks(
        self, chunks: list[Chunk], *, pipeline_run_id: str, document_id: str, version: str
    ) -> EnrichmentArtifact:
        enriched_chunks: list[EnrichedChunk] = []
        all_entities: dict[str, Entity] = {}
        all_relations: dict[tuple[str, str, str, str], Relation] = {}
        all_temporal: dict[tuple[str, str], TemporalAnchor] = {}

        for chunk in chunks:
            entities, relations, temporal, keywords = self.extract_text(chunk.text, evidence_id=chunk.id)
            enriched_chunks.append(
                EnrichedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    version=chunk.version,
                    block_ids=chunk.block_ids,
                    entities=entities,
                    relations=relations,
                    temporal=temporal,
                    keywords=keywords,
                )
            )
            for entity in entities:
                all_entities[entity.normalized_name] = entity
            for relation in relations:
                all_relations[(relation.subject, relation.predicate, relation.object, relation.evidence_id)] = relation
            for anchor in temporal:
                all_temporal[(anchor.normalized, anchor.granularity)] = anchor

        return EnrichmentArtifact(
            pipeline_run_id=pipeline_run_id,
            document_id=document_id,
            version=version,
            chunks=enriched_chunks,
            entities=sorted(all_entities.values(), key=lambda e: e.normalized_name),
            relations=list(all_relations.values()),
            temporal=sorted(all_temporal.values(), key=lambda t: (t.normalized, t.granularity)),
        )
