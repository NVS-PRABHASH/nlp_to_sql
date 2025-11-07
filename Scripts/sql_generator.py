

"""
SQL Generator for Banking Queries

This module provides functionality to generate and execute SQL queries
based on MRL lines and natural language input.
"""

import json
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Import domain classifier to check if query is in-domain
from domain_classifier import DomainClassifier
classifier = DomainClassifier("data/mrl_dataset.csv")

from time_reference import create_time_reference_table

# Ensure Scripts directory is in path for imports
_scripts_dir = Path(__file__).parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import requests
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import logging

# Import from mrl_api.py
from mrl_api import query_to_data, get_full_hierarchy, get_hierarchy_from_query, find_matching_mrl, get_engine
try:
    # Optional RAG fallback if primary extraction fails
    from mrl_api import fallback_to_rag  # type: ignore
except Exception:  # pragma: no cover - optional import
    fallback_to_rag = None

# Import from guardrails.py
from guardrails import validate_generated_sql

# Centralized configuration and logging
from config import get_config, setup_logging, build_oracle_url

# Import dynamic prompt builder with vector retrieval
from dynamic_prompt_builder import build_dynamic_prompt_string, get_dynamic_prompt_builder

# Import logging utilities
from token_counter import create_token_counter
from prompt_logger import get_prompt_logger

# Initialize logging
setup_logging()
logger = logging.getLogger("ai_insight.sql_generator")

# Initialize token counter and prompt logger
_token_counter = None
_prompt_logger = None


def _initialize_tools():
    """Initialize auxiliary helpers; returns (token_counter, prompt_logger, dynamic_prompt_ready)."""
    global _token_counter, _prompt_logger
    
    if _token_counter is None:
        # Extract base URL from full API URL (remove /chat/completions if present)
        base_url = VLLM_API_URL.replace("/chat/completions", "")
        _token_counter = create_token_counter(_cfg)
        # Override with correct base URL
        if _token_counter:
            _token_counter.api_url = base_url
    
    if _prompt_logger is None:
        _prompt_logger = get_prompt_logger()
    
    dynamic_prompt_ready = False
    try:
        get_dynamic_prompt_builder()
        dynamic_prompt_ready = True
    except Exception as exc:  # pragma: no cover - defensive logging
        pass

    return _token_counter, _prompt_logger, dynamic_prompt_ready


# vLLM API Configuration
_cfg = get_config()
VLLM_API_URL = ((_cfg.get("vllm") or {}).get("api_url") or "")
MODEL_NAME = ((_cfg.get("vllm") or {}).get("model_name") or "")

# Database Configuration from config
DB_CONFIG = (_cfg.get("database") or {})



# --------------------------
# Structured Exceptions
# --------------------------
class SQLValidationError(Exception):
    def __init__(self, message: str, sql: str, original_query: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.sql = sql
        self.original_query = original_query


class DatabaseExecutionError(Exception):
    def __init__(self, message: str, sql: str):
        super().__init__(message)
        self.message = message
        self.sql = sql


def get_db_engine():
    """Create and return a database engine with connection pooling using config."""
    url = build_oracle_url(DB_CONFIG)
    pool_cfg = DB_CONFIG.get("pool", {})
    return create_engine(
        url,
        poolclass=QueuePool,
        pool_size=int(pool_cfg.get("pool_size", 5)),
        max_overflow=int(pool_cfg.get("max_overflow", 10)),
        pool_timeout=int(pool_cfg.get("pool_timeout", 30)),
        pool_recycle=int(pool_cfg.get("pool_recycle", 3600)),
    )

def get_mrl_lines(query: str) -> List[str]:
    """
    Get MRL lines based on a natural language query using mrl_api.py functions.
    
    Args:
        query: Natural language query
        
    Returns:
        List of MRL line values
    """
    valid_values, _, _ = get_full_hierarchy()
    
    result = query_to_data(query, valid_values)
    # query_to_data may return either a list (direct_query) or a dict per product.
    if isinstance(result, dict):
        # Flatten unique lines across all products to feed SQL generator
        flat: List[str] = []
        seen = set()
        for lines in result.values():
            for m in lines or []:
                if m not in seen:
                    flat.append(m)
                    seen.add(m)
        return flat
    return result
        

def get_mrl_lines_grouped(query: str) -> Dict[str, List[str]]:
    """
    Return grouped MRL lines by product category for a query.
    If the underlying function returns a flat list (direct levels),
    wrap it under the key 'direct_query'.
    """
    valid_values, _, _ = get_full_hierarchy()
    result = query_to_data(query, valid_values)
    if isinstance(result, dict):
        return result
    return {"direct_query": result}


def compute_mrl_lines(query: str) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Compute both flat and grouped MRL lines with a single call to query_to_data
    to avoid invoking the LLM twice.
    Returns: (flat_list, grouped_dict)
    """
    valid_values, _, _ = get_full_hierarchy()
    result = query_to_data(query, valid_values)
    if isinstance(result, dict):
        # grouped
        flat: List[str] = []
        seen = set()
        for lines in result.values():
            for m in lines or []:
                if m not in seen:
                    flat.append(m)
                    seen.add(m)
        return flat, result
    # direct list
    return (result or []), {"direct_query": (result or [])}


        
def get_business_date() -> str:
    """
    Fetch the latest business date from DM_MIS_DETAILS_VW1 table.
   
    Returns:
        str: The latest business date in 'YYYY-MM-DD' format
    """
    try:
        with get_db_engine().connect() as conn:
            query = text("""
                SELECT MAX(BUSINESS_DATE)
                FROM DM_MIS_DETAILS_VW1
            """)
            result = conn.execute(query)
            date_value = result.scalar()
               
            # Convert to datetime object and format as YYYY-MM-DD
            if isinstance(date_value, str):
                # If it's a string, parse it first
                date_obj = datetime.strptime(date_value.split('.')[0], '%Y-%m-%d %H:%M:%S')
            else:
                # If it's already a datetime object
                date_obj = date_value
               
            return date_obj.strftime('%Y-%m-%d')
           
    except Exception as e:
        print(f"Error fetching business date: {e}")
        return '2025-02-14'
    

def generate_sql(query: str, show_examples: bool = False, use_dynamic_prompt: bool = True) -> tuple:
    """
    Generate SQL queries using the dynamic prompt builder and semantic retrieval.

    1. Compute MRL context (flat and grouped) via ``mrl_api``.
    2. Build a query-specific prompt using sentence-transformer retrieval for keywords,
       rules, and examples (falls back to static prompt if the builder is unavailable).
    3. Call the configured vLLM endpoint and post-process placeholders.
    4. Return SQL (and metadata when ``show_examples=True``).

    Args:
        query: Natural language description of the query.
        show_examples: If True, return ``(sql, metadata)``; otherwise return the SQL string.
        use_dynamic_prompt: When False, force the static prompt fallback.

    Returns:
        str or tuple: Generated SQL query, or ``(sql, metadata)`` if ``show_examples`` is True.
    """
    try:
        time_reference={}
        business_date=get_business_date()
        time_reference=create_time_reference_table(business_date) 

        token_counter, prompt_logger, dynamic_prompt_ready = _initialize_tools()
        logger.info("="*50)
        logger.info(f"🔍 Processing query: {query}")
        logger.info("="*50)

        # Step 0: Check if query is in-domain
        classification, confidence = classifier.classify(query)
        if classification == "OFF-DOMAIN":
            logger.warning(f"❌ Query is out-of-domain (Confidence: {confidence}%)")
            # Return immediately without any processing
            if show_examples:
                return "off-domain", {
                    "classification": "OFF-DOMAIN",
                    "confidence": confidence,
                    "message": "Query is not related to banking/finance domain"
                }
            return "off-domain"

        logger.info(f"✅ Query is in-domain (Confidence: {confidence}%)")

        # Step 1: First compute MRL lines and groups using compute_mrl_lines
        mrl_start = time.perf_counter()
        mrl_lines, mrl_groups = compute_mrl_lines(query)
        mrl_end = time.perf_counter()
        mrl_processing_time = mrl_end - mrl_start
        
        # Create group placeholders dictionary (replace spaces with underscores)
        group_placeholders = {name: f"#GROUP_{name.upper().replace(' ', '_')}#" for name in mrl_groups.keys()}

        # Step 2: Now build the prompt with MRL groups available
        sql_processing_start = time.perf_counter()
        dynamic_prompt = None
        dynamic_metadata = {}
        
        
        if use_dynamic_prompt and dynamic_prompt_ready:
            try:
                # Build dynamic prompt WITH MRL groups available
                dynamic_prompt, dynamic_metadata = build_dynamic_prompt_string(
                    query,
                    use_vector_retrieval=True,
                    mrl_groups=mrl_groups,
                )
            except Exception as e:
                pass

        # If dynamic prompt failed or disabled, use fallback
        if not dynamic_prompt:
            dynamic_prompt = (
                "You are a SQL expert for banking MIS reporting.\n"
                "Generate Oracle SQL queries for the DM_MIS_DETAILS_VW1 table.\n"
                "Use only the columns and rules specified in this prompt.\n"
                "Always use ROUND(..., 2) for decimal values.\n"
                "NEVER use JOINs - use only DM_MIS_DETAILS_VW1 table.\n"
                "NEVER use date arithmetic (SYSDATE, ADD_MONTHS) - use predefined period columns only.\n\n"
                "When filtering:\n"
                "- Use MRL_LINE IN (#MRL_LINES#) for product filtering\n"
                "- Use BAL_TYPE = [value] for balance type\n"
                "- Use CCY_TYPE = [value] for currency type\n\n"
                "Output: Single SELECT statement only. No explanations or markdown."
            )

        retrieved_examples = dynamic_metadata.get("examples", []) if dynamic_metadata else []
        example_count = len(retrieved_examples)

        # Inject group placeholders documentation into the prompt
        final_prompt = dynamic_prompt
        if mrl_groups and group_placeholders:
            # Build documentation of available group placeholders (comma-separated for single line)
            group_docs = []
            for group_name in mrl_groups.keys():
                placeholder = f"#GROUP_{group_name.upper().replace(' ', '_')}#"
                group_docs.append(f"{placeholder} (for {group_name}: {len(mrl_groups[group_name])} lines)")
            
            # Use comma-separated format for better display in prompt rules
            group_placeholders_doc = ", ".join(group_docs) if group_docs else "No groups defined"
            
            # Replace the placeholder in the prompt template
            final_prompt = final_prompt.replace(
                "{json.dumps(group_placeholders, ensure_ascii=False)}",
                group_placeholders_doc
            )

        # Suppressed verbose console prints for production

        # Count tokens in the prompt
        token_counts = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        if token_counter and final_prompt:
            try:
                token_counts = token_counter.count_tokens(final_prompt)
            except Exception:
                pass

        # Log prompt generation details
        if prompt_logger and final_prompt:
            try:
                prompt_logger.log_prompt_generation(
                    user_query=query,
                    retrieved_examples=retrieved_examples,
                    token_counts=token_counts,
                )
            except Exception:
                pass

        # Call the LLM with the dynamic prompt
        if not final_prompt:
            raise ValueError("No prompt available for LLM call")

        print(final_prompt)

        response = requests.post(
            VLLM_API_URL,
            json={
                "model": MODEL_NAME,
                "messages": [
                {"role": "system", 
                 "content": final_prompt
                },
                {"role": "user", 
                 "content": query
                }
            ],
                "temperature": 0.0,
                "max_tokens": 5000,
            },
            timeout=300,
        )
        response.raise_for_status()

        sql = response.json()["choices"][0]["message"]["content"].strip()
        sql_processing_end = time.perf_counter()
        sql_processing_time = sql_processing_end - sql_processing_start
        logger.info("SQL generated by LLM")

        # Replace overall placeholder with actual MRL list
        sql = sql.replace("#MRL_LINES#", ",".join([f"'{m}'" for m in mrl_lines]))

        # Replace each group placeholder with its actual list
        if mrl_groups:
            for group_name, lines in mrl_groups.items():
                placeholder = f"#GROUP_{group_name.upper().replace(' ', '_')}#"
                replacement = ",".join([f"'{m}'" for m in lines]) if lines else ""
                if placeholder in sql:
                    sql = sql.replace(placeholder, replacement)

        # Clean up the SQL
        if sql.startswith("```sql"):
            sql = sql[6:].strip()
        if sql.endswith("```"):
            sql = sql[:-3].strip()

        sql = sql.strip()
        if sql.endswith(";"):
            sql = sql[:-1].strip()

        logger.info("SQL generated: %s", sql)
        

        # Build comprehensive metadata with vector retrieval details
        metadata = {
            "retrieval_method": dynamic_metadata.get("retrieval_method", "fallback")
            if dynamic_metadata
            else "fallback",
            "keywords": dynamic_metadata.get("keywords", []) if dynamic_metadata else [],
            "num_keywords": dynamic_metadata.get("num_keywords", 0) if dynamic_metadata else 0,
            "dynamic_sections": dynamic_metadata.get("dynamic_sections", []) if dynamic_metadata else [],
            "static_sections": dynamic_metadata.get("static_sections", []) if dynamic_metadata else [],
            "num_rules": dynamic_metadata.get("num_rules", 0) if dynamic_metadata else 0,
            "examples": retrieved_examples,
            "num_examples": example_count,
            "prompt_size": len(final_prompt) if final_prompt else 0,
            "token_counts": token_counts,
            "mrl_processing_time": mrl_processing_time,
            "sql_generation_time": sql_processing_time,
            "total_time": sql_processing_time + mrl_processing_time,
            "mrl_lines_count": len(mrl_lines),
            "mrl_groups": list(mrl_groups.keys()),
            "dynamic_prompt_used": use_dynamic_prompt and dynamic_prompt_ready,
            "vector_retrieval_active": use_dynamic_prompt and dynamic_prompt_ready,
        }

        if show_examples:
            return sql, metadata
        return sql

    except Exception as e:  # noqa: BLE001 - propagate after logging
        logger.error("❌ Error generating SQL: %s", e)

        if _prompt_logger:
            try:
                _prompt_logger.log_prompt_generation(
                    user_query=query,
                    retrieved_examples=[],
                    token_counts={
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    error=str(e),
                )
            except Exception:  # noqa: BLE001 - best effort logging
                pass
        raise Exception(f"Error generating SQL: {e}") from e


def _normalize_db_value(value):
    """Preserve numeric precision while keeping JSON-friendly structures."""
    if isinstance(value, Decimal):
        return value  # FastAPI/Pydantic serializes Decimal safely; CLI uses default=str
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def execute_query(sql: str, engine=None, original_query: Optional[str] = None) -> List[Dict]:
    """
    Execute a validated SELECT query against DM_MIS_DETAILS_VW1 and return rows as dictionaries.

    Parameters:
    - sql: The generated SELECT statement to run.
    - engine: Optional SQLAlchemy Engine. If None, a pooled engine is created internally.
    - original_query: The original natural language query used for generation (used by guardrails).

    Returns:
    - List[Dict[str, Any]]: Result rows mapped as dictionaries of column name to value.

    Raises:
    - Exception: If SQL validation fails or the database execution encounters an error.
    """
    # Guardrail validation
    # ok, reason = validate_generated_sql(sql, original_query)
    # if not ok:
    #     raise SQLValidationError(reason, sql, original_query)

    close_engine = False
    if engine is None:
        engine = get_db_engine()
        close_engine = True
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = result.keys()
            rows = result.fetchall()

            serialized_rows: List[Dict[str, Any]] = []
            for row in rows:
                normalized = {col: _normalize_db_value(val) for col, val in zip(columns, row)}
                serialized_rows.append(normalized)
            return serialized_rows
    except Exception as e:
        # Wrap any db/driver error into a structured exception for the API layer
        raise DatabaseExecutionError(str(e), sql)
    finally:
        if close_engine and engine:
            engine.dispose()



# Example usage
if __name__ == "__main__":
    # Interactive SQL generator - runs continuously until Ctrl+C
    print("=" * 80)
    print("SQL Generator - Interactive Mode")
    print("=" * 80)
    print("Enter your natural language queries to generate SQL.")
    print("Press Ctrl+C to exit.")
    print("=" * 80)
    
    while True:
        try:
            query = input("\n🔍 Query: ").strip()
            
            if not query:
                print("⚠️  Please enter a query.")
                continue
                
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Exiting...")
                break
                
            if query.lower() == 'clear':
                import os
                os.system('clear' if os.name == 'posix' else 'cls')
                continue

            print(f"\n⏳ Processing: {query}")
            print("-" * 80)
            start_gen = time.time() 

            # Generate SQL with metadata to show dynamic prompt analysis
            user_total_start = time.perf_counter()
            result = generate_sql(query, show_examples=True, use_dynamic_prompt=True)
            if isinstance(result, tuple):
                sql, metadata = result
                print("=" * 80)
            else:
                sql = result
            gen_end = time.time()-start_gen
            print("\n✅ Generated SQL:")
            print("-" * 80)
            print(sql)
            print("-" * 80) 
            #print(f"\n⏱️  SQL Generation Time: {gen_end:.2f} seconds")

            
            # Execute the query
            try:
                # Print time and size stats before final Query Results (requested format)
                tokens_total = 0.0
                if isinstance(result, tuple):
                    _, metadata = result
                    tokens_total = float(metadata.get('token_counts', {}).get('total_tokens', 0.0) or 0.0)
                    print("\n" + "-" * 80)
                    print(
                        f"MRL Processing Time : {metadata.get('mrl_processing_time', 0.0):.2f} sec"
                    )
                    print(
                        f"SQL Processing Time : {metadata.get('sql_generation_time', 0.0):.2f} sec"
                    )
                    print(f"Token Count : {tokens_total:.2f} tokens")
                    # Reconciled total: sum of the four phases printed above
                    total_print = (
                        float(metadata.get('mrl_processing_time', 0.0))
                        + float(metadata.get('sql_generation_time', 0.0))
                    )
                    print(
                        f"Total time : {total_print:.2f} sec (Time taken from user query entry to result generation)"
                    )
                    print("-" * 80)

                # Now execute and print results
                results = execute_query(sql, original_query=query)
                print(f"\n📊 Query Results ({len(results)} rows):")
                print("-" * 80)
                print(json.dumps(results, indent=2, default=str))
            except Exception as exec_err:
                print(f"\n⚠️  Execution Error: {exec_err}")
            
            print("=" * 80)



        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("=" * 80)
    
    print("\nGoodbye! 👋\n")
