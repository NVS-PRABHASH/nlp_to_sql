#!/usr/bin/env python3
"""
Thread Manager for Conversation Memory

This module manages thread lifecycle including thread ID generation,
turn ID generation, and thread title creation.
"""

import logging
import uuid
import requests
from typing import Optional
from datetime import datetime

from config import get_config, setup_logging

# Initialize logging
setup_logging()
logger = logging.getLogger("ai_insight.thread_manager")


class ThreadManager:
    """Manager class for conversation thread operations."""

    def __init__(self):
        """Initialize thread manager with configuration."""
        self.config = get_config()
        self.conversation_config = self.config.get("conversation", {})
        self.vllm_config = self.config.get("vllm", {})
        
        self.api_url = self.vllm_config.get("api_url", "")
        self.model_name = self.vllm_config.get("model_name", "")
        self.title_max_length = self.conversation_config.get("thread_title_max_length", 50)
        self.title_temperature = self.conversation_config.get("title_generation_temperature", 0.3)

    def generate_thread_id(self) -> str:
        """
        Generate a unique thread ID.
        
        Returns:
            Unique thread identifier string
        """
        # Use UUID with timestamp for uniqueness
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        thread_id = f"thread_{timestamp}_{unique_id}"
        
        logger.info(f"Generated new thread_id: {thread_id}")
        return thread_id

    def generate_turn_id(self, thread_id: str) -> str:
        """
        Generate a unique turn ID for a conversation turn.
        
        Args:
            thread_id: Thread identifier
            
        Returns:
            Unique turn identifier string
        """
        # Use UUID with timestamp for uniqueness
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")[:17]  # Include microseconds
        unique_id = uuid.uuid4().hex[:6]
        turn_id = f"{thread_id}_turn_{timestamp}_{unique_id}"
        
        logger.debug(f"Generated turn_id: {turn_id}")
        return turn_id

    def create_thread_title(self, first_query: str) -> str:
        """
        Generate a descriptive thread title from the first query using LLM.
        
        Args:
            first_query: The first user query in the thread
            
        Returns:
            Descriptive thread title (max length from config)
        """
        try:
            # Prepare prompt for title generation
            system_prompt = """You are a title generator for banking query conversations.
Generate a clear, descriptive title (3-7 words) that captures the main banking topic or product mentioned in the query.

Examples:
- "What are total deposits in retail banking?" → "Retail Banking Deposits Analysis"
- "Show me term loan balances" → "Term Loan Balance Report"
- "Interest income from corporate clients" → "Corporate Interest Income"
- "NPA provisions for last quarter" → "NPA Provisions Analysis"

Keep it concise, professional, and banking-domain specific."""

            user_prompt = f"""Generate a descriptive title for this banking query:

Query: {first_query}

Title (3-7 words):"""

            # Call vLLM API
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": self.title_temperature,
                    "max_tokens": 50,
                },
                timeout=10,
            )
            
            if response.status_code == 200:
                result = response.json()
                title = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                
                # Clean up the title
                title = title.strip('"').strip("'").strip()
                
                # Limit to max length
                if len(title) > self.title_max_length:
                    title = title[:self.title_max_length].rsplit(' ', 1)[0] + "..."
                
                logger.info(f"Generated thread title: {title}")
                return title
            else:
                logger.warning(f"LLM API returned status {response.status_code}, using fallback title")
                return self._fallback_title(first_query)
        
        except Exception as e:
            logger.error(f"Error generating thread title with LLM: {e}")
            return self._fallback_title(first_query)

    def _fallback_title(self, query: str) -> str:
        """
        Generate a fallback title when LLM call fails.
        
        Args:
            query: User query
            
        Returns:
            Simple fallback title
        """
        # Extract key banking terms if present
        banking_terms = [
            "deposit", "loan", "interest", "npa", "balance", "income",
            "asset", "liability", "provision", "retail", "corporate",
            "term loan", "advance", "investment"
        ]
        
        query_lower = query.lower()
        found_terms = [term for term in banking_terms if term in query_lower]
        
        if found_terms:
            # Capitalize first letters
            title = " ".join([term.title() for term in found_terms[:3]])
            if len(title) > self.title_max_length:
                title = title[:self.title_max_length].rsplit(' ', 1)[0] + "..."
            return title
        else:
            # Generic fallback
            return "Banking Query"


# Singleton instance
_thread_manager = None


def get_thread_manager() -> ThreadManager:
    """Get or create singleton ThreadManager instance."""
    global _thread_manager
    if _thread_manager is None:
        _thread_manager = ThreadManager()
    return _thread_manager


if __name__ == "__main__":
    # Test the thread manager
    print("Testing Thread Manager...")
    
    manager = get_thread_manager()
    
    # Test thread ID generation
    thread_id = manager.generate_thread_id()
    print(f"✓ Generated thread_id: {thread_id}")
    
    # Test turn ID generation
    turn_id = manager.generate_turn_id(thread_id)
    print(f"✓ Generated turn_id: {turn_id}")
    
    # Test title generation
    test_query = "What are the total deposits in retail banking for last quarter?"
    title = manager.create_thread_title(test_query)
    print(f"✓ Generated title: {title}")
    
    print("\nThread manager is ready!")
