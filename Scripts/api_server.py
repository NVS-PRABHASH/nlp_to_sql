#!/usr/bin/env python3
"""
FastAPI Server for Banking Query Processing

This module provides a RESTful API for processing natural language banking queries
and returning MRL lines or SQL query results.
"""

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from httpx import post
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn
import logging
import os

# Import existing functionality
from mrl_api import get_engine, get_full_hierarchy, query_to_data
from sql_generator import generate_sql, execute_query, SQLValidationError, DatabaseExecutionError

# Import conversation memory
from conversation_memory import get_conversation_memory

# Centralized config and logging
from config import setup_logging, get_config

# Configure logging using config
setup_logging()
logger = logging.getLogger("ai_insight.api")

# Initialize FastAPI app
app = FastAPI(
    title="Banking Query Processing API",
    description="API for processing natural language banking queries and generating SQL",
    version="1.0.0"
)

# Configure CORS from config
cfg = get_config()
cors = ((cfg.get("server") or {}).get("cors") or {})

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors.get("allow_origins", ["*"]),
    allow_credentials=cors.get("allow_credentials", True),
    allow_methods=cors.get("allow_methods", ["*"]),
    allow_headers=cors.get("allow_headers", ["*"]),
)

# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    query: str
    user_id: str  # Required: User identifier for conversation tracking
    thread_id: Optional[str] = None  # Optional: Existing thread to continue
    context: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    status: int  # 1 for success, 0 for failure
    sql: str
    thread_id: Optional[str] = None  # NEW: Thread identifier
    turn_id: Optional[str] = None  # NEW: Turn identifier
    message: Optional[str] = None
    results: Optional[Any] = None  # Can be str, dict, or list
    metadata: Optional[Dict[str, Any]] = None

class NewThreadRequest(BaseModel):
    user_id: str

class DeleteThreadRequest(BaseModel):
    user_id: str
    thread_id: str

class ThreadsRequest(BaseModel):
    user_id: str
    limit: Optional[int] = 10

class HistoryRequest(BaseModel):
    user_id: str
    thread_id: str
    limit: Optional[int] = 50

# Database connection dependency
async def get_db():
    engine = None
    try:
        engine = get_engine()
        yield engine
    except Exception as e:
        # Do not raise here; let handlers return a uniform failed response
        logger.error(f"Database connection error: {str(e)}")
        yield None
    finally:
        if engine:
            engine.dispose()

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


# Create new thread
@app.post("/sql/threads/new")
async def create_new_thread(request: NewThreadRequest):
    """
    Create a new thread for a user without saving to database.
    Thread will be saved on first query.
    
    Args:
        request: NewThreadRequest with user_id
        
    Returns:
        JSON with status (1=success, 0=error), thread_id, and message
    """
    try:
        from thread_manager import get_thread_manager
        
        thread_manager = get_thread_manager()
        
        # Generate new thread_id (no DB write)
        thread_id = thread_manager.generate_thread_id()
        
        logger.info(f"Generated new thread_id for user {request.user_id}: {thread_id}")
        
        return {
            "status": 1,
            "thread_id": thread_id,
            "message": "New thread created"
        }
    
    except Exception as e:
        logger.error(f"Error creating new thread: {e}")
        return {
            "status": 0,
            "thread_id": None,
            "message": f"Failed to create thread: {str(e)}"
        }


# Delete thread
@app.delete("/sql/threads/delete")
async def delete_thread(request: DeleteThreadRequest):
    """
    Delete a thread and all its conversation history.
    
    Args:
        request: DeleteThreadRequest with user_id and thread_id
        
    Returns:
        JSON with status (1=success, 0=error) and message
    """
    try:
        from qdrant_service import get_qdrant_service
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        qdrant_service = get_qdrant_service()
        
        # First verify thread belongs to user
        history = qdrant_service.get_thread_history(
            user_id=request.user_id,
            thread_id=request.thread_id,
            limit=1
        )
        
        if not history:
            # Thread doesn't exist or doesn't belong to user
            logger.warning(f"Thread {request.thread_id} not found for user {request.user_id}")
            return {
                "status": 0,
                "message": "Thread not found or does not belong to user"
            }
        
        # Delete all points with matching thread_id and user_id
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=request.user_id),
                ),
                FieldCondition(
                    key="thread_id",
                    match=MatchValue(value=request.thread_id),
                ),
            ]
        )
        
        # Perform deletion
        result = qdrant_service.client.delete(
            collection_name=qdrant_service.collection_name,
            points_selector=filter_condition
        )
        
        logger.info(f"Deleted thread {request.thread_id} for user {request.user_id}")
        
        return {
            "status": 1,
            "message": "Thread deleted successfully"
        }
    
    except Exception as e:
        logger.error(f"Error deleting thread: {e}")
        return {
            "status": 0,
            "message": f"Failed to delete thread: {str(e)}"
        }


# Get user's conversation threads
@app.post("/sql/threads")
async def get_threads(request: ThreadsRequest):
    """
    Retrieve user's recent conversation threads with titles.
    
    Args:
        request: ThreadsRequest with user_id and optional limit
        
    Returns:
        JSON with threads array containing thread_id and title
    """
    try:
        memory = get_conversation_memory()
        
        if not memory.enabled:
            return {
                "status": 1,
                "threads": []
            }
        
        result = memory.get_user_threads(request.user_id, request.limit)
        result["status"] = 1
        return result
    
    except Exception as e:
        logger.error(f"Error retrieving user threads: {e}")
        return {
            "status": 0,
            "threads": [],
            "message": f"Failed to retrieve threads: {str(e)}"
        }


# Get specific thread chat history
@app.post("/sql/history")
async def get_thread_history(request: HistoryRequest):
    """
    Retrieve chat history for a specific thread.
    
    Args:
        request: HistoryRequest with user_id, thread_id, and optional limit
        
    Returns:
        JSON with full conversation history in chronological order (oldest first)
        
    Response Format (Success):
        {
            "status": "success",
            "user_id": "test_123",
            "thread_id": "thread_20251104154030_f011bc51",
            "thread_title": "Retail Banking Deposits Analysis",
            "created_at": "2025-11-04T15:40:30.123456",
            "last_updated": "2025-11-04T15:41:32.654321",
            "history": [
                {
                    "turn_id": "...",
                    "timestamp": "...",
                    "query": "What are the total deposits?",
                    "sql": "SELECT ...",
                    "results": {...}
                }
            ],
            "total_turns": 2,
            "sort_order": "asc"
        }
        
    Response Format (Unauthorized):
        {
            "status": "error",
            "error_code": "unauthorized",
            "message": "Thread does not belong to user"
        }
    """
    try:
        memory = get_conversation_memory()
        
        if not memory.enabled:
            return {
                "status": 0,
                "message": "Conversation memory is disabled"
            }
        
        # Get thread history
        history_result = memory.get_thread_history(request.user_id, request.thread_id, request.limit)
        history = history_result.get("history", [])
        
        # Check if thread exists and belongs to user
        if not history:
            # Check if thread exists for another user (authorization check)
            # Try to get thread with no user filter to see if it exists
            from qdrant_service import get_qdrant_service
            qdrant = get_qdrant_service()
            
            # Get all user's threads to verify ownership
            user_threads = qdrant.get_user_threads(request.user_id, limit=1000)
            thread_exists_for_user = any(t.get("thread_id") == request.thread_id for t in user_threads)
            
            if not thread_exists_for_user:
                # Thread might exist but for different user or doesn't exist at all
                return {
                    "status": 0,
                    "message": f"Thread {request.thread_id} does not belong to user {request.user_id} or does not exist"
                }
            else:
                # Thread exists for user but no history (edge case)
                return {
                    "status": 1,
                    "user_id": request.user_id,
                    "thread_id": request.thread_id,
                    "thread_title": "Empty Thread",
                    "created_at": None,
                    "last_updated": None,
                    "history": [],
                    "total_turns": 0,
                    "sort_order": "asc"
                }
        
        # Extract thread metadata
        thread_title = None
        created_at = None
        last_updated = None
        
        # Find thread title from history (first turn should have it)
        for turn in history:
            if turn.get("thread_title"):
                thread_title = turn["thread_title"]
                break
        
        if not thread_title:
            thread_title = "Untitled Thread"
        
        # Sort history chronologically (oldest first)
        sorted_history = sorted(history, key=lambda x: x.get("timestamp", ""))
        
        # Get created_at (oldest) and last_updated (newest)
        if sorted_history:
            created_at = sorted_history[0].get("timestamp")
            last_updated = sorted_history[-1].get("timestamp")
        
        # Format history for response (clean up the data structure)
        formatted_history = []
        for turn in sorted_history:
            formatted_turn = {
                "turn_id": turn.get("turn_id"),
                "timestamp": turn.get("timestamp"),
                "query": turn.get("query"),
                "sql": turn.get("response"),  # 'response' field contains SQL
                "results": turn.get("results")
            }
            formatted_history.append(formatted_turn)
        
        return {
            "status": 1,
            "user_id": request.user_id,
            "thread_id": request.thread_id,
            "thread_title": thread_title,
            "created_at": created_at,
            "last_updated": last_updated,
            "history": formatted_history,
            "total_turns": len(formatted_history),
            "sort_order": "asc"
        }
    
    except Exception as e:
        logger.error(f"Error retrieving thread history: {e}")
        return {
            "status": 0,
            "message": f"Failed to retrieve thread history: {str(e)}"
        }


# Generate SQL Query and return
@app.post("/api/sql", response_model=QueryResponse)
async def generate_sql_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    engine=Depends(get_db)
):
    """
    Generate SQL from a natural language banking query.
    Includes conversation memory tracking with thread management.
    """
    try:
        if engine is None:
            return QueryResponse(
                status=0, 
                sql="", 
                thread_id=None,
                turn_id=None,
                message="database_unavailable", 
                results={"error": "database_unavailable"}
            )

        # Generate SQL using the existing functionality
        result = generate_sql(query=request.query, show_examples=True, use_dynamic_prompt=True)
        if isinstance(result, tuple):
            sql_text, metadata = result
        else:
            sql_text = result
            metadata = None
        
        # Check if query is off-domain
        if sql_text == "off-domain":
            logger.info(f"Query rejected as off-domain: {request.query}")
            return QueryResponse(
                status=0,
                sql="off-domain",
                thread_id=None,
                turn_id=None,
                message="off-domain",
                results={"error": "off-domain", "message": "Query is not related to banking/finance domain"},
                metadata=metadata
            )
        
        # Execute the query to get results
        results = None
        try:
            results = execute_query(sql=sql_text, engine=engine, original_query=request.query)
        except Exception as exec_error:
            logger.warning(f"Query execution failed, storing SQL anyway: {exec_error}")
            results = {"error": "execution_failed"}
        
        # Process conversation memory in background (non-blocking)
        thread_id = "disabled"
        turn_id = "disabled"
        
        try:
            memory = get_conversation_memory()
            if memory.enabled:
                # Process synchronously to get thread_id and turn_id for response
                # But upsert can be done in background if needed
                thread_id, turn_id = memory.process_query(
                    user_id=request.user_id,
                    query=request.query,
                    response_sql=sql_text,
                    results=results,
                    thread_id=request.thread_id,
                )
        except Exception as mem_error:
            # Graceful error handling - don't fail the request if memory fails
            logger.error(f"Conversation memory error: {mem_error}")
            thread_id = "error"
            turn_id = "error"
        
        return QueryResponse(
            status=1, 
            sql=sql_text, 
            thread_id=thread_id,
            turn_id=turn_id,
            message="success", 
            metadata=metadata
        )

    except SQLValidationError as e:
        logger.error(f"SQL validation error: {e.message}\nSQL: {e.sql}")
        return QueryResponse(
            status=0, 
            sql="", 
            thread_id=None,
            turn_id=None,
            message="SQL Validation Error"
        )
    except DatabaseExecutionError as e:
        logger.error(f"Database execution error: {e.message}\nSQL: {e.sql}")
        return QueryResponse(
            status=0, 
            sql="", 
            thread_id=None,
            turn_id=None,
            message="Invalid Query. Try another Query"
        )
    except Exception as e:
        logger.error(f"Unexpected error executing SQL: {str(e)}")
        return QueryResponse(
            status=0, 
            sql="", 
            thread_id=None,
            turn_id=None,
            message=str(e), 
            results={"error": str(e)}
        )

# Execute SQL query and return results
#@app.post("/api/execute", response_model=QueryResponse)
#async def execute_sql_query(
#    request: QueryRequest,
#    engine=Depends(get_db)
#):
#    """
#    Execute a SQL query and return the results.
#    """
#    try:
#        if engine is None:
#            return QueryResponse(status=0, sql="", message="database_unavailable", results={"error": "database_unavailable"})
#
#        if not request.query:
#            return QueryResponse(status=0, sql="", message="empty_query", results={"error": "empty_query"})
#
#        raw = request.query.strip()
#        # Always generate SQL from the provided query (treat as NL)
#        sql_text = generate_sql(query=raw)
#        logger.info(f"Generated SQL from NL query: {sql_text}")
#
#        results = execute_query(sql=sql_text, engine=engine, original_query=raw)
#
#        return QueryResponse(status=1, sql=sql_text, message="success", results=results)
#
#    except SQLValidationError as e:
#        logger.error(f"SQL validation error: {e.message}\nSQL: {e.sql}")
#        return QueryResponse(status=0, sql="", message="SQL Validation Error", results={"error": "sql_validation_error"})
#    except DatabaseExecutionError as e:
#        logger.error(f"Database execution error: {e.message}\nSQL: {e.sql}")
#        return QueryResponse(status=0, sql="", message="Invalid Query. Try another Query", results={"error": "invalid_query"})
#    except Exception as e:
#        logger.error(f"Unexpected error executing SQL: {str(e)}")
#        return QueryResponse(status=0, sql="", message="Internal Error", results={"error": "internal_error"})

if __name__ == "__main__":
    # Run the FastAPI application using config
    server = cfg.get("server", {}) or {}

    host = server.get("host", "0.0.0.0")
    port = int(server.get("port", 5005))
    reload_ = bool(server.get("reload", True))
    workers = int(server.get("workers", 4))  # keep 1 when reload=True
    

    # HTTPS settings (absolute paths recommended)
    ssl_certfile = server.get("ssl_certfile", "/home/clouddeployment/deployment/server.crt")
    ssl_keyfile  = server.get("ssl_keyfile",  "/home/clouddeployment/deployment/server.key")
    print("CERT:", ssl_certfile, os.path.isfile(ssl_certfile))
    print("KEY:", ssl_keyfile, os.path.isfile(ssl_keyfile))
    
    # Ensure reload & workers aren't used together
    if reload_ and workers != 1:
        logger.warning("`reload` and `workers` are mutually exclusive. Forcing workers=1 because reload=True.")
        workers = 1
    
    # Check if conversation memory is enabled with multiple workers
    conversation_config = cfg.get("conversation", {})
    if conversation_config.get("enable_memory", True) and workers > 1:
        logger.warning("⚠️  Conversation memory enabled with workers > 1. Local Qdrant storage requires workers=1.")
        logger.warning("⚠️  Forcing workers=1 to prevent 'storage already accessed' errors.")
        workers = 1

    # Log which mode we’re in
    if os.path.isabs(ssl_certfile) and os.path.isfile(ssl_certfile) and \
       os.path.isabs(ssl_keyfile)  and os.path.isfile(ssl_keyfile):
        logger.info(f"Starting with HTTPS on {host}:{port}")
        uvicorn.run(
            "api_server:app",
            host=host,
            port=port,
            reload=reload_,
            workers=workers,
            
        )
    else:
        logger.warning("SSL cert/key not found or not absolute; starting without TLS. "
                       "Set server.ssl_certfile and server.ssl_keyfile to absolute paths.")
        uvicorn.run(
            "api_server:app",
            host=host,
            port=port,
            reload=reload_,
            workers=workers,
        )


