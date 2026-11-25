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
    "who", "why", "will", "with", "would", "you", "your",
    # Vietnamese
    "va", "cua", "la", "cac", "co", "duoc", "cho", "trong", "mot", "nhung", "nay",
    "khi", "voi", "tu", "den", "tai", "ve", "nhu", "khong", "da", "se", "dang", "theo",
    "qua", "sau", "truoc", "de", "neu", "thi", "hay", "hoac", "boi", "vi", "nen",
    "và", "của", "là", "các", "có", "được", "cho", "trong", "một", "những", "này",
    "với", "từ", "đến", "tại", "về", "như", "không", "đã", "sẽ", "đang", "để", "nếu",
    "thì", "hoặc", "bởi", "vì", "nên", "trên", "dưới", "giữa", "cùng", "cũng", "vẫn",
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
    "bun": ("Bun", "technology"),
    "nginx": ("Nginx", "technology"),
    "bm25": ("BM25", "technology"),
    "mcp": ("MCP", "technology"),
    "rag": ("RAG", "concept"),
    "llm": ("LLM", "concept"),
    "openai": ("OpenAI", "organization"),
    "github": ("GitHub", "organization"),
    "memory lab": ("Memory Lab", "concept"),
    "vectormemory": ("VectorMemory", "technology"),
    "graphmemory": ("GraphMemory", "technology"),
    "hybridmemory": ("HybridMemory", "technology"),
    "zerotokenmemory": ("ZeroTokenMemory", "technology"),
}

MONTH_MAP = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

RELATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2})\s+(?:uses|using|use)\s+([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2})\b"), "uses"),
    (re.compile(r"\b([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2})\s+(?:depends on|relies on)\s+([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2})\b"), "depends_on"),
    (re.compile(r"\b([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2})\s+(?:connects to|integrates with)\s+([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2})\b"), "integrates_with"),
    (re.compile(r"\b([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2})\s+(?:stores|indexes|persists)\s+([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2})\b"), "stores"),
    (re.compile(r"\b([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2})\s+(?:created|built|authored|maintains)\s+([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2})\b"), "created"),
    (re.compile(r"\b([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2})\s+(?:is a|is an|extends|implements)\s+([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2})\b"), "is_a"),
    (re.compile(r"\b([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2})\s+(?:sử dụng|dùng)\s+([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2})\b"), "uses"),
    (re.compile(r"\b([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2})\s+(?:lưu trữ|lưu)\s+([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2})\b"), "stores"),
    (re.compile(r"\b([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2})\s+(?:kết nối với|kết nối)\s+([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2})\b"), "integrates_with"),
    (re.compile(r"\b([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2})\s+(?:phụ thuộc vào|phụ thuộc)\s+([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2})\b"), "depends_on"),
]


def fold_ascii(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).replace("đ", "d").replace("Đ", "D")


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).lower()


def tokenize(text: str, *, include_folded: bool = True) -> list[str]:
    raw_tokens = re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)
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
            if name in {"HTTP", "JSON", "UUID", "UTF", "URL", "URI", "GET", "POST", "PUT", "DELETE"}:
                etype = "technology"
            else:
                etype = "organization"
            norm = normalize_name(name)
            if norm not in found and norm not in STOPWORDS:
                found[norm] = Entity(name=name, normalized_name=norm, type=etype)

        # 4. Multi-word capitalized proper noun phrases
        for match in re.finditer(r"\b[A-ZÀ-Ỹ][a-zà-ỹ0-9]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ0-9]+){1,3}\b", text):
            name = match.group(0).strip()
            norm = normalize_name(name)
            if norm not in found and norm not in STOPWORDS:
                found[norm] = Entity(name=name, normalized_name=norm, type="concept")

        # 5. Capitalized mid-sentence single proper nouns (not at start of sentence)
        for sentence in re.split(r"[.!?\n]+", text):
            words = sentence.strip().split()
            for raw_word in words[1:]:
                clean = raw_word.strip(",:;()[]{}\"'")
                if re.fullmatch(r"[A-Z][a-z]{2,}", clean):
                    norm = normalize_name(clean)
                    if norm not in found and norm not in STOPWORDS and norm not in MONTH_MAP:
                        found[norm] = Entity(name=clean, normalized_name=norm, type="concept")

        return sorted(found.values(), key=lambda e: e.normalized_name)

    def extract_relations(self, text: str, entities: list[Entity], evidence_id: str = "") -> list[Relation]:
        relations: dict[tuple[str, str, str], Relation] = {}

        # Explicit pattern-based relations
        for pattern, predicate in RELATION_PATTERNS:
            for match in pattern.finditer(text):
                subj = normalize_name(match.group(1))
                obj = normalize_name(match.group(2))
                if subj and obj and subj != obj and subj not in STOPWORDS and obj not in STOPWORDS:
                    key = (subj, predicate, obj)
                    relations[key] = Relation(
                        subject=subj, predicate=predicate, object=obj, evidence_id=evidence_id, confidence=0.9
                    )

        # Sentence-level entity co-occurrence relations (capped to avoid noise)
        entity_norms = [e.normalized_name for e in entities[:12]]
        if len(entity_norms) >= 2:
            for sentence in re.split(r"[.!?\n]+", text):
                lower_sentence = sentence.lower()
                present = [e for e in entity_norms if re.search(rf"(?<!\w){re.escape(e)}(?!\w)", lower_sentence)]
                for i in range(min(len(present), 5)):
                    for j in range(i + 1, min(len(present), 5)):
                        subj, obj = sorted((present[i], present[j]))
                        if not any(r.subject == subj and r.object == obj for r in relations.values()):
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
        tokens = [
            tok for tok in tokenize(text, include_folded=False)
            if tok not in STOPWORDS and fold_ascii(tok) not in STOPWORDS and not tok.isdigit()
        ]
        counts = Counter(tokens)
        return [word for word, _count in counts.most_common(limit)]

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
