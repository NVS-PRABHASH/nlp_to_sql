# # --- SQL Guardrails ---
# import re
# from typing import Optional
# ALLOWED_TABLE = "DM_MIS_DETAILS_VW1"

# FORBIDDEN_KEYWORDS = [
#     ";", "--", "/*", "*/", " BEGIN", " END", " COMMIT", " ROLLBACK",
#     " INSERT ", " UPDATE ", " DELETE ", " MERGE ", " DROP ", " ALTER ", " TRUNCATE ",
#     " CREATE ", " GRANT ", " REVOKE ", " EXEC ", " EXECUTE ", " CALL ", " DBMS_"
# ]

# def _only_allowed_table(sql: str) -> bool:
#     s = sql.upper()
#     if re.search(r"\bJOIN\b", s):
#         return False
#     m = re.search(r"\bFROM\b\s+([A-Z0-9_.]+)", s)
#     if not m:
#         return False
#     table = m.group(1)
#     return table.split(".")[-1] == ALLOWED_TABLE

# def _contains_forbidden_keywords(sql: str) -> tuple[bool, str]:
#     s = " " + sql.upper() + " "
#     for kw in FORBIDDEN_KEYWORDS:
#         if kw in s:
#             return True, f"Forbidden keyword detected: {kw.strip()}"
#     return False, ""

# def _extract_where(sql: str) -> str:
#     s = sql.upper()
#     m = re.search(r"\bWHERE\b(.*)$", s, re.DOTALL)
#     return m.group(1) if m else ""

# def _is_net_income_or_profit(query_text: Optional[str]) -> bool:
#     if not query_text:
#         return False
#     t = query_text.upper()
#     return ("NET INCOME" in t) or ("NET PROFIT" in t)

# def _has_mandatory_filters(sql: str, require_mrl: bool) -> tuple[bool, str]:
#     where = _extract_where(sql)
#     if not where:
#         return False, "Missing WHERE clause"
#     if require_mrl and "MRL_LINE" not in where:
#         return False, "Missing MRL_LINE filter"
#     if "BAL_TYPE" not in where:
#         return False, "Missing BAL_TYPE filter"
#     if "CCY_TYPE" not in where:
#         return False, "Missing CCY_TYPE filter"
#     return True, ""

# def validate_generated_sql(sql: str, original_query: Optional[str] = None) -> tuple[bool, str]:
#     if not sql or not sql.strip().upper().startswith("SELECT"):
#         return False, "Only SELECT statements are allowed"
#     bad, reason = _contains_forbidden_keywords(sql)
#     if bad:
#         return False, reason
#     if not _only_allowed_table(sql):
#         return False, "Only DM_MIS_DETAILS_VW1 without JOINs is allowed"
#     # Require MRL_LINE except when the original query is net income/profit
#     require_mrl = not _is_net_income_or_profit(original_query)
#     ok, reason = _has_mandatory_filters(sql, require_mrl=require_mrl)
#     if not ok:
#         return False, reason
#     return True, ""

# --- SQL Guardrails (self-contained) ---
import re
from typing import Optional, Tuple

ALLOWED_TABLE = "DM_MIS_DETAILS_VW1"

FORBIDDEN_KEYWORDS = [
    ";", "--", "/*", "*/", " BEGIN", " COMMIT", " ROLLBACK",
    " INSERT ", " UPDATE ", " DELETE ", " MERGE ", " DROP ", " ALTER ", " TRUNCATE ",
    " CREATE ", " GRANT ", " REVOKE ", " EXEC ", " EXECUTE ", " CALL ", " DBMS_"
]

def _only_allowed_table(sql: str) -> bool:
    s = sql.upper()
    if re.search(r"\bJOIN\b", s):
        return False
    m = re.search(r"\bFROM\b\s+([A-Z0-9_.]+)", s)
    if not m:
        return False
    table = m.group(1)
    return table.split(".")[-1] == ALLOWED_TABLE

def _contains_forbidden_keywords(sql: str) -> Tuple[bool, str]:
    s = " " + sql.upper() + " "
    for kw in FORBIDDEN_KEYWORDS:
        if kw in s:
            return True, f"Forbidden keyword detected: {kw.strip()}"
    return False, ""

def _extract_where(sql: str) -> str:
    s = sql.upper()
    m = re.search(r"\bWHERE\b(.*)$", s, re.DOTALL)
    return m.group(1) if m else ""

def _is_net_income_or_profit(query_text: Optional[str]) -> bool:
    if not query_text:
        return False
    t = query_text.upper()
    return ("NET INCOME" in t) or ("NET PROFIT" in t)

def _has_mandatory_filters(sql: str, require_mrl: bool) -> Tuple[bool, str]:
    where = _extract_where(sql)
    if not where:
        return False, "Missing WHERE clause"
    if require_mrl and "MRL_LINE" not in where:
        return False, "Missing MRL_LINE filter (required for this query type)"
    if "BAL_TYPE" not in where:
        return False, "Missing BAL_TYPE filter"
    if "CCY_TYPE" not in where:
        return False, "Missing CCY_TYPE filter"
    return True, ""

def validate_generated_sql(sql: str, original_query: Optional[str] = None) -> Tuple[bool, str]:
    """Validate generated SQL against security and business rules without external deps."""
    if not sql or not isinstance(sql, str):
        return False, "Empty SQL"

    sql_stripped = sql.strip()
    if not sql_stripped.upper().startswith("SELECT"):
        return False, "Only SELECT statements are allowed"

    if not _only_allowed_table(sql_stripped):
        return False, f"Only queries against {ALLOWED_TABLE} are allowed"

    bad, reason = _contains_forbidden_keywords(sql_stripped)
    if bad:
        return False, reason

    # Net income/profit queries don't need MRL_LINE
    require_mrl = not _is_net_income_or_profit(original_query)
    ok, reason = _has_mandatory_filters(sql_stripped, require_mrl)
    if not ok:
        return False, reason

    return True, ""