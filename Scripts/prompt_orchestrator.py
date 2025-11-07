# ai_insight/Scripts/prompt_orchestrator.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import math, re, hashlib, os, json
from collections import Counter

def _load_system_prompt() -> str:
    """Load system prompt from system_prompt.txt at repo root; fallback to default."""
    here = os.path.dirname(__file__)
    path = os.path.join(here, "system_prompt.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            return txt if txt else "You are an expert Oracle SQL generator for banking analytics."
    except Exception:
        return (
            "You are an expert Oracle SQL generator for banking analytics.\n"
            "You write production-grade ORACLE SQL only.\n"
            "Output exactly one SELECT statement. No explanations, markdown, comments, or prefixes."
        )

# System prompt is now loaded from the external file
SYSTEM_PROMPT = _load_system_prompt()

# ---------- Minimal vector index (TF-IDF cosine) ----------
class SimpleVectorIndex:
    def __init__(self, texts: List[str], ids: Optional[List[str]] = None):
        self.texts = texts
        self.ids = ids or [f"doc_{i}" for i in range(len(texts))]
        self.df = Counter()
        self.N = len(texts)
        self.vocabs: List[Dict[str, float]] = []
        self._build()

    @staticmethod
    def _tokenize(s: str) -> List[str]:
        return re.findall(r"[a-z0-9_]+", s.lower())

    def _build(self):
        docs_tokens = [self._tokenize(t) for t in self.texts]
        for tokens in docs_tokens:
            for term in set(tokens):
                self.df[term] += 1
        self.vocabs = []
        for tokens in docs_tokens:
            tf = Counter(tokens)
            vec = {}
            for term, f in tf.items():
                idf = math.log((1 + self.N) / (1 + self.df[term])) + 1.0
                vec[term] = (f / len(tokens)) * idf
            self.vocabs.append(vec)

    @staticmethod
    def _cos(a: Dict[str, float], b: Dict[str, float]) -> float:
        num = 0.0
        for t, v in a.items():
            if t in b:
                num += v * b[t]
        da = math.sqrt(sum(v * v for v in a.values())) or 1e-12
        db = math.sqrt(sum(v * v for v in b.values())) or 1e-12
        return num / (da * db)

    def embed(self, text: str) -> Dict[str, float]:
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        vec = {}
        for term, f in tf.items():
            idf = math.log((1 + self.N) / (1 + self.df.get(term, 0))) + 1.0
            vec[term] = (f / (len(tokens) or 1)) * idf
        return vec

    def search(self, query: str, k: int = 3) -> List[tuple]:
        qv = self.embed(query)
        scores = []
        for i, dv in enumerate(self.vocabs):
            s = self._cos(qv, dv)
            scores.append((self.ids[i], s, self.texts[i]))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

# ---------- Prompt types ----------
@dataclass
class PromptSnippet:
    id: str
    text: str
    tags: List[str] = field(default_factory=list)

@dataclass
class PromptLibrary:
    snippets: List[PromptSnippet] = field(default_factory=list)
    _index: Optional[SimpleVectorIndex] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "PromptLibrary":
        items = []
        for rec in data.get("snippets", []):
            items.append(PromptSnippet(id=rec["id"], text=rec["text"], tags=rec.get("tags", [])))
        lib = cls(items)
        lib._build_index()
        return lib

    def _build_index(self):
        texts = [s.text for s in self.snippets]
        ids = [s.id for s in self.snippets]
        self._index = SimpleVectorIndex(texts, ids)

    def topk(self, query: str, k: int = 2, tag_filter: Optional[List[str]] = None) -> List[PromptSnippet]:
        if not self._index or not self.snippets:
            return []
        hits = self._index.search(query, k=10)
        out: List[PromptSnippet] = []
        for doc_id, _, _ in hits:
            snip = next((s for s in self.snippets if s.id == doc_id), None)
            if not snip:
                continue
            if tag_filter and not any(t in snip.tags for t in tag_filter):
                continue
            out.append(snip)
            if len(out) >= k:
                break
        return out

# ---------- Classification ----------
class QueryClassifier:
    INVALID_PAT = re.compile(r"\b(drop|alter|truncate|insert|update|delete)\b", re.I)
    OFFDOMAIN_PAT = re.compile(r"\b(weather|movie|song|football|stock price)\b", re.I)
    SENSITIVE_PAT = re.compile(r"\b(ssn|social security|password|credential)\b", re.I)

    @staticmethod
    def classify(q: str) -> str:
        if not q or len(q.strip()) < 5:
            return "too_vague"
        if QueryClassifier.SENSITIVE_PAT.search(q):
            return "sensitive"
        if QueryClassifier.INVALID_PAT.search(q):
            return "invalid"
        if QueryClassifier.OFFDOMAIN_PAT.search(q):
            return "off_domain"
        return "valid"

# ---------- Prompt Builder ----------
@dataclass
class BuildConfig:
    model_name: str = "Qwen/Qwen2.5-14B-Instruct"
    max_examples: int = 2
    allowed_tables: Dict[str, List[str]] = field(default_factory=dict)
    oracle_date_col_hints: List[str] = field(default_factory=lambda: ["OPEN_DATE","BALANCE_DATE","TXN_DATE"])

class PromptBuilder:
    DEV_TMPL = """Constraints:
- Dialect: Oracle 12c+ (TRUNC(date,'MM'), ADD_MONTHS, EXTRACT, NVL, FETCH FIRST n ROWS ONLY).
- Output: single SELECT only. No DDL/DML. No multiple statements.
- Safety: Use ONLY allowed tables/columns below. If a requested field is missing, return exactly: /*NO_ANSWER*/
- Aggregations require proper GROUP BY. Avoid SELECT *.
- Dates: month grain uses TRUNC({date_col}, 'MM'). Quarters via TRUNC(SYSDATE,'Q').
Allowed schema:
{allowed_schema}
"""

    @staticmethod
    def render_allowed_schema(allowed: Dict[str, List[str]]) -> str:
        if not allowed:
            return "- CORE.CUSTOMER(...)\n- CORE.ACCOUNT(...)\n- CORE.TXN(...)"
        return "\n".join([f"- {t}({', '.join(cols)})" for t, cols in allowed.items()])

    @staticmethod
    def pack_messages(system: str, developer: str, examples: List[str], user: str) -> List[Dict[str, str]]:
        msgs = [{"role": "system", "content": system}, {"role": "developer", "content": developer}]
        msgs.extend({"role": "developer", "content": ex} for ex in examples)
        msgs.append({"role": "user", "content": user})
        return msgs

    def build(self, q: str, lib: PromptLibrary, cfg: BuildConfig) -> tuple:
        examples = [s.text for s in lib.topk(q, k=cfg.max_examples, tag_filter=None)]
        allowed = self.render_allowed_schema(cfg.allowed_tables)
        date_col = cfg.oracle_date_col_hints[0] if cfg.oracle_date_col_hints else "OPEN_DATE"
        developer = self.DEV_TMPL.format(allowed_schema=allowed, date_col=date_col)
        user = f"User request:\n{q.strip()}\n"
        msgs = self.pack_messages(SYSTEM_PROMPT, developer, examples, user)
        meta = {"examples_used": len(examples), "prompt_hash": hashlib.md5(("|".join(m["content"] for m in msgs)).encode()).hexdigest()}
        return msgs, meta

# ---------- Output validation ----------
class OutputGuard:
    MULTISTMT = re.compile(r";\s*\S")
    FENCE = re.compile(r"```")
    COMMENT = re.compile(r"--|/\*")
    DML_DDL = re.compile(r"\b(insert|update|delete|merge|create|alter|drop|truncate)\b", re.I)

    @staticmethod
    def is_sql_safe(sql: str) -> bool:
        if OutputGuard.FENCE.search(sql): return False
        if OutputGuard.COMMENT.search(sql): return False
        if OutputGuard.MULTISTMT.search(sql): return False
        if OutputGuard.DML_DDL.search(sql): return False
        if not re.search(r"^\s*select\b", sql, re.I): return False
        return True

# ---------- Orchestrator ----------
class PromptOrchestrator:
    def __init__(self, lib: PromptLibrary, cfg: BuildConfig):
        self.lib = lib
        self.cfg = cfg
        self.builder = PromptBuilder()

    def route(self, user_query: str) -> Dict:
        cls = QueryClassifier.classify(user_query or "")
        if cls != "valid":
            return {
                "status": "short_path",
                "class": cls,
                "messages": [],
                "decision": "NO_LLM",
                "output": "/*NO_ANSWER*/" if cls in {"invalid","sensitive","off_domain"} else "/*NEED_MORE_DETAIL*/"
            }
        msgs, meta = self.builder.build(user_query, self.lib, self.cfg)
        return {"status":"ready","class":cls,"messages":msgs,"decision":"CALL_LLM","meta":meta}

    def validate_output(self, sql: str) -> tuple:
        safe = OutputGuard.is_sql_safe(sql or "")
        return safe, (sql if safe else "/*NO_ANSWER*/")

# ---------- Convenience: Assemble final prompt as a string ----------
def _pack_messages_to_string(messages: List[Dict[str, str]]) -> str:
    """Turn chat messages into a single string with role separators for inspection/logging."""
    parts: List[str] = []
    for i, m in enumerate(messages, 1):
        role = m.get("role", "").upper()
        content = m.get("content", "")
        parts.append(f"--- {i}. {role} ---\n{content}")
    return "\n\n".join(parts)

def _load_prompt_library() -> PromptLibrary:
    """Load prompt library from prompt_lib.json if present; else return empty library."""
    here = os.path.dirname(__file__)
    pj = os.path.join(here, "prompt_lib.json")
    data: Dict = {"snippets": []}
    try:
        if os.path.isfile(pj):
            with open(pj, "r", encoding="utf-8") as f:
                data = json.load(f) or {"snippets": []}
    except Exception:
        data = {"snippets": []}
    return PromptLibrary.from_dict(data)

def _load_allowed_schema() -> Dict[str, List[str]]:
    """Load allowed schema from allowed_schema.json if present; else provide minimal placeholders."""
    here = os.path.dirname(__file__)
    sj = os.path.join(here, "allowed_schema.json")
    try:
        if os.path.isfile(sj):
            with open(sj, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    # Fallback minimal schema
    return {
        "CORE.CUSTOMER": ["CUSTOMER_ID","CUSTOMER_NAME","CITY","STATE","SEGMENT","OPEN_DATE"],
        "CORE.ACCOUNT":  ["ACCOUNT_ID","CUSTOMER_ID","PRODUCT","BALANCE_AMT","INTEREST_RATE","CCY","OPEN_DATE"],
        "CORE.TXN":      ["TXN_ID","ACCOUNT_ID","TXN_DATE","TXN_AMT","TXN_TYPE"],
    }

def generate_sql(query: str) -> str:
    """
    Build the vectorized prompt (system + developer + examples + user) and return the
    entire final prompt as a single string. This does NOT call any LLM.

    Args:
        query: user natural-language request

    Returns:
        str: Full prompt text suitable for inspection or sending to a single-turn model.
    """
    lib = _load_prompt_library()
    allowed = _load_allowed_schema()
    cfg = BuildConfig(allowed_tables=allowed)
    builder = PromptBuilder()
    messages, _meta = builder.build(query, lib, cfg)
    return _pack_messages_to_string(messages)


def generate_sql_with_metadata(query: str) -> tuple:
    """
    Build the vectorized prompt and return both the prompt string and metadata
    including retrieved examples.

    Args:
        query: user natural-language request

    Returns:
        tuple: (prompt_string, metadata_dict)
            - prompt_string: Full prompt text
            - metadata_dict: Contains 'examples' (list of dicts with 'id' and 'text'),
                            'examples_count', 'prompt_hash'
    """
    lib = _load_prompt_library()
    allowed = _load_allowed_schema()
    cfg = BuildConfig(allowed_tables=allowed)
    builder = PromptBuilder()
    
    # Get the top-k examples directly from the library
    retrieved_snippets = lib.topk(query, k=cfg.max_examples, tag_filter=None)
    
    # Build messages
    messages, meta = builder.build(query, lib, cfg)
    
    # Enhance metadata with example details
    meta['examples'] = [
        {'id': snip.id, 'text': snip.text, 'tags': snip.tags}
        for snip in retrieved_snippets
    ]
    
    prompt_string = _pack_messages_to_string(messages)
    return prompt_string, meta
