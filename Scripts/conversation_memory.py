#!/usr/bin/env python3
"""
Conversation Memory Orchestrator

This module orchestrates all conversation memory features including
thread management, context checking, and vector database operations.
"""

import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

from qdrant_service import get_qdrant_service
from thread_manager import get_thread_manager
from context_checker import get_context_checker
from config import get_config, setup_logging

# Initialize logging
setup_logging()
logger = logging.getLogger("ai_insight.conversation_memory")


class ConversationMemory:
    """Main orchestrator for conversation memory features."""

    def __init__(self):
        """Initialize conversation memory with all required services."""
        self.config = get_config()
        self.conversation_config = self.config.get("conversation", {})
        
        # Feature flag
        self.enabled = self.conversation_config.get("enable_memory", True)
        
        if self.enabled:
            # Initialize services
            self.qdrant_service = get_qdrant_service()
            self.thread_manager = get_thread_manager()
            self.context_checker = get_context_checker()
            
            # Configuration
            self.default_tenant_id = self.conversation_config.get("default_tenant_id", 1)
            self.max_thread_history = self.conversation_config.get("max_thread_history", 10)
            
            logger.info("Conversation memory initialized (enabled)")
        else:
            logger.info("Conversation memory disabled by configuration")

    def process_query(
        self,
        user_id: str,
        query: str,
        response_sql: str,
        results: Any,
        thread_id: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Process a new query and manage conversation memory.
        
        This is the main entry point that:
        1. Determines thread_id (new or existing)
        2. Checks relevance if thread exists
        3. Generates turn_id
        4. Creates thread title if new thread
        5. Upserts to Qdrant
        
        Args:
            user_id: User identifier
            query: User's natural language query
            response_sql: Generated SQL query
            results: Query execution results
            thread_id: Optional existing thread_id (client can force thread continuation)
            
        Returns:
            Tuple of (thread_id, turn_id)
        """
        if not self.enabled:
            logger.debug("Conversation memory disabled, skipping")
            return "disabled", "disabled"
        
        try:
            # Step 1: Determine thread_id
            is_new_thread = False
            thread_title = None
            
            if thread_id:
                # Client provided thread_id, check if it exists and is relevant
                logger.info(f"Client provided thread_id: {thread_id}")
                
                # Get last k queries in this thread
                previous_queries = self.qdrant_service.get_last_k_queries_in_thread(
                    user_id=user_id,
                    thread_id=thread_id,
                    k=self.max_thread_history,
                )
                
                if previous_queries:
                    # Thread exists, check relevance
                    is_relevant, reason = self.context_checker.check_with_max_history(
                        current_query=query,
                        previous_queries=previous_queries,
                        max_history=self.max_thread_history,
                    )
                    
                    if not is_relevant:
                        # Query not relevant, create new thread
                        logger.info(f"Query not relevant to thread {thread_id}, creating new thread. Reason: {reason}")
                        thread_id = self.thread_manager.generate_thread_id()
                        is_new_thread = True
                    else:
                        logger.info(f"Query relevant to thread {thread_id}, continuing thread")
                else:
                    # Thread doesn't exist or no history, treat as new thread
                    logger.info(f"Thread {thread_id} has no history, treating as new thread")
                    is_new_thread = True
            else:
                # No thread_id provided - check if user has previous queries
                logger.info("No thread_id provided, checking user's query history")
                
                # Get user's most recent thread
                most_recent_thread_id = self.qdrant_service.get_most_recent_thread(user_id)
                
                if most_recent_thread_id:
                    # User has previous queries, check relevance to most recent thread (last 5 queries)
                    logger.info(f"User has previous queries, checking relevance to most recent thread: {most_recent_thread_id}")
                    
                    previous_queries = self.qdrant_service.get_last_k_queries_in_thread(
                        user_id=user_id,
                        thread_id=most_recent_thread_id,
                        k=5,  # Check only the last 5 queries as specified
                    )
                    
                    if previous_queries:
                        # Check if current query is relevant to the most recent thread
                        is_relevant, reason = self.context_checker.check_with_max_history(
                            current_query=query,
                            previous_queries=previous_queries,
                            max_history=5,
                        )
                        
                        if is_relevant:
                            # Continue the most recent thread
                            thread_id = most_recent_thread_id
                            logger.info(f"Query relevant to most recent thread {thread_id}, continuing thread")
                        else:
                            # Create new thread
                            logger.info(f"Query not relevant to most recent thread, creating new thread. Reason: {reason}")
                            thread_id = self.thread_manager.generate_thread_id()
                            is_new_thread = True
                    else:
                        # Most recent thread has no queries (shouldn't happen), create new
                        logger.warning(f"Most recent thread {most_recent_thread_id} has no queries, creating new thread")
                        thread_id = self.thread_manager.generate_thread_id()
                        is_new_thread = True
                else:
                    # First query from this user, create new thread
                    logger.info("First query from user, creating new thread")
                    thread_id = self.thread_manager.generate_thread_id()
                    is_new_thread = True
            
            # Step 2: Generate turn_id
            turn_id = self.thread_manager.generate_turn_id(thread_id)
            
            # Step 3: Handle thread title
            if is_new_thread:
                # Generate title for new thread
                thread_title = self.thread_manager.create_thread_title(query)
                logger.info(f"New thread created: {thread_id} | Title: {thread_title}")
            else:
                # Get existing title for continued thread
                thread_title = self.qdrant_service.get_thread_title(user_id, thread_id)
                if thread_title:
                    logger.info(f"Continuing thread {thread_id} with title: {thread_title}")
                else:
                    logger.warning(f"Thread {thread_id} has no title, will use None")
            
            # Step 4: Upsert to Qdrant
            success = self.qdrant_service.upsert_conversation(
                query=query,
                response=response_sql,
                results=results,
                tenant_id=self.default_tenant_id,
                user_id=user_id,
                thread_id=thread_id,
                turn_id=turn_id,
                thread_title=thread_title,
            )
            
            if success:
                logger.info(f"Successfully stored conversation: user={user_id}, thread={thread_id}, turn={turn_id}")
            else:
                logger.error(f"Failed to store conversation in Qdrant")
            
            return thread_id, turn_id
        
        except Exception as e:
            logger.error(f"Error processing conversation memory: {e}")
            # Return graceful fallback
            return "error", "error"

    def get_user_threads(self, user_id: str, limit: int = 10) -> Dict[str, Any]:
        """
        Retrieve user's recent conversation threads.
        
        Args:
            user_id: User identifier
            limit: Maximum number of threads to return
            
        Returns:
            Dictionary with threads array
        """
        if not self.enabled:
            return {"threads": []}
        
        try:
            threads = self.qdrant_service.get_user_threads(user_id, limit)
            
            # Format response
            formatted_threads = [
                {
                    "thread_id": t["thread_id"],
                    "title": t["title"],
                }
                for t in threads
            ]
            
            logger.info(f"Retrieved {len(formatted_threads)} threads for user {user_id}")
            return {"threads": formatted_threads}
        
        except Exception as e:
            logger.error(f"Error retrieving user threads: {e}")
            return {"threads": []}

    def get_thread_history(
        self,
        user_id: str,
        thread_id: str,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Retrieve conversation history for a specific thread.
        
        Args:
            user_id: User identifier
            thread_id: Thread identifier
            limit: Maximum number of turns to return
            
        Returns:
            Dictionary with conversation history
        """
        if not self.enabled:
            return {"history": []}
        
        try:
            history = self.qdrant_service.get_thread_history(user_id, thread_id, limit)
            
            logger.info(f"Retrieved {len(history)} turns for thread {thread_id}")
            return {"history": history}
        
        except Exception as e:
            logger.error(f"Error retrieving thread history: {e}")
            return {"history": []}


# Singleton instance
_conversation_memory = None


def get_conversation_memory() -> ConversationMemory:
    """Get or create singleton ConversationMemory instance."""
    global _conversation_memory
    if _conversation_memory is None:
        _conversation_memory = ConversationMemory()
    return _conversation_memory


if __name__ == "__main__":
    # Test the conversation memory orchestrator
    print("Testing Conversation Memory...")
    
    memory = get_conversation_memory()
    print(f"✓ Conversation memory initialized (enabled: {memory.enabled})")
    
    if memory.enabled:
        # Test process_query
        print("\n--- Test 1: First Query (New Thread) ---")
        thread_id, turn_id = memory.process_query(
            user_id="test_user_123",
            query="What are the total deposits in retail banking?",
            response_sql="SELECT SUM(amount) FROM deposits WHERE category='retail'",
            results={"total": 1000000},
            thread_id=None,
        )
        print(f"✓ Thread ID: {thread_id}")
        print(f"✓ Turn ID: {turn_id}")
        
        # Test process_query with existing thread
        print("\n--- Test 2: Related Query (Same Thread) ---")
        thread_id2, turn_id2 = memory.process_query(
            user_id="test_user_123",
            query="Break down deposits by branch",
            response_sql="SELECT branch, SUM(amount) FROM deposits WHERE category='retail' GROUP BY branch",
            results=[{"branch": "A", "total": 500000}, {"branch": "B", "total": 500000}],
            thread_id=thread_id,
        )
        print(f"✓ Thread ID: {thread_id2}")
        print(f"✓ Turn ID: {turn_id2}")
        print(f"✓ Same thread: {thread_id == thread_id2}")
        
        # Test get_user_threads
        print("\n--- Test 3: Get User Threads ---")
        threads = memory.get_user_threads("test_user_123", limit=10)
        print(f"✓ Retrieved {len(threads['threads'])} threads")
        for t in threads['threads']:
            print(f"  - {t['thread_id']}: {t['title']}")
    
    print("\nConversation memory is ready!")
