#!/usr/bin/env python3
"""
Qdrant Vector Database Service

This module provides integration with Qdrant for storing and retrieving
conversation history with semantic search capabilities.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer

from config import get_config, setup_logging

# Initialize logging
setup_logging()
logger = logging.getLogger("ai_insight.qdrant_service")


class QdrantService:
    """Service class for managing Qdrant vector database operations."""
    
    # Class-level shared client and embedder (singleton pattern)
    _shared_client = None
    _shared_embedder = None
    _storage_path = None

    def __init__(self):
        """Initialize Qdrant client and embedding model using singleton pattern."""
        self.config = get_config()
        self.qdrant_config = self.config.get("qdrant", {})
        self.conversation_config = self.config.get("conversation", {})
        
        # Initialize Qdrant client (local disk storage) - SINGLETON
        storage_path = self.qdrant_config.get("storage_path", "./qdrant_data")
        self.collection_name = self.qdrant_config.get("collection_name", "conversation_history")
        
        # Resolve storage path relative to project root
        if not Path(storage_path).is_absolute():
            project_root = Path(__file__).parent.parent
            storage_path = project_root / storage_path
        
        # Use shared client if already initialized for same storage path
        if QdrantService._shared_client is None or QdrantService._storage_path != str(storage_path):
            logger.info(f"Creating NEW Qdrant client with storage: {storage_path}")
            QdrantService._shared_client = QdrantClient(path=str(storage_path))
            QdrantService._storage_path = str(storage_path)
        else:
            logger.info(f"✓ Reusing existing Qdrant client (singleton pattern)")
        
        self.client = QdrantService._shared_client
        
        # Initialize embedding model (singleton)
        if QdrantService._shared_embedder is None:
            embedding_model = self.qdrant_config.get(
                "embedding_model", 
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info(f"Loading embedding model: {embedding_model}")
            QdrantService._shared_embedder = SentenceTransformer(embedding_model)
        else:
            logger.debug(f"Reusing existing embedding model")
        
        self.embedder = QdrantService._shared_embedder
        
        # Initialize collection
        self._initialize_collection()

    def _initialize_collection(self):
        """Create Qdrant collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                vector_size = self.qdrant_config.get("vector_size", 384)
                distance = self.qdrant_config.get("distance", "Cosine")
                
                # Map string distance to Qdrant Distance enum
                distance_map = {
                    "Cosine": Distance.COSINE,
                    "Euclidean": Distance.EUCLID,
                    "Dot": Distance.DOT,
                }
                distance_metric = distance_map.get(distance, Distance.COSINE)
                
                logger.info(f"Creating collection '{self.collection_name}' with vector_size={vector_size}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=distance_metric
                    ),
                )
                logger.info(f"Collection '{self.collection_name}' created successfully")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
        
        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {e}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for given text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            embedding = self.embedder.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def upsert_conversation(
        self,
        query: str,
        response: str,
        results: Any,
        tenant_id: int,
        user_id: str,
        thread_id: str,
        turn_id: str,
        thread_title: Optional[str] = None,
    ) -> bool:
        """
        Insert or update a conversation turn in Qdrant.
        
        Args:
            query: User's natural language query
            response: Generated SQL query
            results: Query execution results
            tenant_id: Tenant identifier
            user_id: User identifier
            thread_id: Thread identifier
            turn_id: Turn identifier (unique per Q&A pair)
            thread_title: Optional thread title
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding from query (for semantic search)
            embedding = self.generate_embedding(query)
            
            # Prepare payload
            payload = {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "response": response,
                "results": results,
            }
            
            # Add thread title if provided
            if thread_title:
                payload["thread_title"] = thread_title
            
            # Create point
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload,
            )
            
            # Upsert to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )
            
            logger.info(f"Upserted conversation: user={user_id}, thread={thread_id}, turn={turn_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error upserting conversation to Qdrant: {e}")
            return False

    def get_thread_history(
        self,
        user_id: str,
        thread_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history for a specific thread.
        
        Args:
            user_id: User identifier
            thread_id: Thread identifier
            limit: Maximum number of turns to retrieve
            
        Returns:
            List of conversation turns ordered by timestamp
        """
        try:
            # Build filter for user_id and thread_id
            filter_conditions = Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id),
                    ),
                    FieldCondition(
                        key="thread_id",
                        match=MatchValue(value=thread_id),
                    ),
                ]
            )
            
            # Search with filter (using a dummy vector since we're filtering, not searching semantically)
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_conditions,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            
            # Extract payloads and sort by timestamp
            conversations = []
            for point in results[0]:  # results is a tuple (points, next_page_offset)
                payload = point.payload
                conversations.append(payload)
            
            # Sort by timestamp (most recent first)
            conversations.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            # Return only the requested limit (in case scroll returned more)
            return conversations[:limit]
        
        except Exception as e:
            logger.error(f"Error retrieving thread history from Qdrant: {e}")
            return []

    def get_user_threads(
        self,
        user_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve unique threads for a user with their titles and metadata.
        
        Args:
            user_id: User identifier
            limit: Maximum number of threads to retrieve
            
        Returns:
            List of threads with thread_id, title, and last_updated timestamp
        """
        try:
            # Build filter for user_id
            filter_conditions = Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id),
                    ),
                ]
            )
            
            # Scroll through all user's conversations
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_conditions,
                limit=1000,  # Retrieve up to 1000 turns to aggregate threads
                with_payload=True,
                with_vectors=False,
            )
            
            # Aggregate threads
            threads_dict = {}
            for point in results[0]:
                payload = point.payload
                thread_id = payload.get("thread_id")
                timestamp = payload.get("timestamp", "")
                thread_title = payload.get("thread_title")  # Can be None
                
                if thread_id:
                    if thread_id not in threads_dict:
                        threads_dict[thread_id] = {
                            "thread_id": thread_id,
                            "title": thread_title if thread_title else "Untitled Thread",
                            "last_updated": timestamp,
                            "turn_count": 0,
                        }
                    
                    # Update last_updated if this turn is more recent
                    if timestamp > threads_dict[thread_id]["last_updated"]:
                        threads_dict[thread_id]["last_updated"] = timestamp
                    
                    # Update title if a valid title exists in this turn (prioritize real titles)
                    if thread_title and threads_dict[thread_id]["title"] == "Untitled Thread":
                        threads_dict[thread_id]["title"] = thread_title
                    
                    threads_dict[thread_id]["turn_count"] += 1
            
            # Convert to list and sort by last_updated (most recent first)
            threads = list(threads_dict.values())
            threads.sort(key=lambda x: x.get("last_updated", ""), reverse=True)
            
            # Return only the requested limit
            return threads[:limit]
        
        except Exception as e:
            logger.error(f"Error retrieving user threads from Qdrant: {e}")
            return []

    def get_last_k_queries_in_thread(
        self,
        user_id: str,
        thread_id: str,
        k: int = 10,
    ) -> List[str]:
        """
        Retrieve the last k queries in a thread for context checking.
        
        Args:
            user_id: User identifier
            thread_id: Thread identifier
            k: Number of last queries to retrieve
            
        Returns:
            List of query strings (most recent first)
        """
        try:
            history = self.get_thread_history(user_id, thread_id, limit=k)
            # Extract only queries
            queries = [turn.get("query", "") for turn in history]
            return queries
        
        except Exception as e:
            logger.error(f"Error retrieving last k queries: {e}")
            return []

    def get_most_recent_thread(self, user_id: str) -> Optional[str]:
        """
        Get the most recent thread_id for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Most recent thread_id or None if user has no threads
        """
        try:
            threads = self.get_user_threads(user_id, limit=1)
            if threads:
                return threads[0].get("thread_id")
            return None
        
        except Exception as e:
            logger.error(f"Error getting most recent thread: {e}")
            return None

    def get_thread_title(self, user_id: str, thread_id: str) -> Optional[str]:
        """
        Get the title of a specific thread.
        
        Args:
            user_id: User identifier
            thread_id: Thread identifier
            
        Returns:
            Thread title or None if not found
        """
        try:
            # Get thread history and find the turn with a title
            history = self.get_thread_history(user_id, thread_id, limit=100)
            
            for turn in history:
                title = turn.get("thread_title")
                if title:
                    return title
            
            return None
        
        except Exception as e:
            logger.error(f"Error getting thread title: {e}")
            return None


# Singleton instance
_qdrant_service = None


def get_qdrant_service() -> QdrantService:
    """Get or create singleton QdrantService instance."""
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
    return _qdrant_service


if __name__ == "__main__":
    # Test the service
    print("Testing Qdrant Service...")
    
    service = get_qdrant_service()
    print(f"✓ Qdrant service initialized")
    print(f"✓ Collection: {service.collection_name}")
    print(f"✓ Storage: {service.qdrant_config.get('storage_path')}")
    
    # Test embedding generation
    test_text = "What are the total deposits in retail banking?"
    embedding = service.generate_embedding(test_text)
    print(f"✓ Generated embedding with dimension: {len(embedding)}")
    
    print("\nQdrant service is ready!")
