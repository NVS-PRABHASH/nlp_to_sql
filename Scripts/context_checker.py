#!/usr/bin/env python3
"""
Context Checker for Thread Relevance

This module uses LangChain and LLM to determine if a new query is
contextually relevant to existing queries in a conversation thread.
Uses strict banking-domain relevance checking.
"""

import logging
import requests
from typing import List, Tuple

from config import get_config, setup_logging

# Initialize logging
setup_logging()
logger = logging.getLogger("ai_insight.context_checker")


class SimpleLLM:
    """Simple LLM wrapper for vLLM API calls."""
    
    def __init__(self, api_url: str, model_name: str, temperature: float = 0.1, max_tokens: int = 200, timeout: int = 15):
        self.api_url = api_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
    
    def call(self, prompt: str, stop: List[str] = None) -> str:
        """Call the vLLM API."""
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stop": stop or [],
                },
                timeout=self.timeout,
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return content.strip()
            else:
                logger.error(f"vLLM API returned status {response.status_code}")
                return "ERROR"
        
        except Exception as e:
            logger.error(f"Error calling vLLM API: {e}")
            return "ERROR"


class ContextChecker:
    """Checker class for determining thread relevance of queries."""

    def __init__(self):
        """Initialize context checker with SimpleLLM."""
        self.config = get_config()
        self.conversation_config = self.config.get("conversation", {})
        self.vllm_config = self.config.get("vllm", {})
        
        # Initialize simple LLM
        self.llm = SimpleLLM(
            api_url=self.vllm_config.get("api_url", ""),
            model_name=self.vllm_config.get("model_name", ""),
            temperature=self.conversation_config.get("relevance_temperature", 0.1),
            max_tokens=200,
            timeout=15,
        )
        
        # Define relevance checking prompt template (as string)
        self.relevance_prompt_template = """You are an expert at analyzing banking and financial queries for contextual relevance.

Your task: Determine if the CURRENT query is contextually related to the PREVIOUS queries in a strict banking-domain sense.

STRICT RELEVANCE RULES:
1. Queries are RELATED if they discuss the same banking product/metric (e.g., deposits, loans, NPA, interest)
2. Queries are RELATED if they refine or drill-down on previous topics (e.g., "show retail only" after "show all deposits")
3. Queries are RELATED if they add time/filter constraints to the same topic (e.g., "last quarter" after "deposits")
4. Queries are RELATED if the product remain same but time period changes (e.g., "this year" after "last year" or "some quarter")
5. Queries are NOT RELATED if they switch to completely different products or topics
6. Queries are NOT RELATED if they're meta-questions like "what was my first question?"
7. Queries are NOT RELATED if it's the user's first question

PREVIOUS QUERIES (most recent first):
{previous_queries}

CURRENT QUERY:
{current_query}

ANALYSIS:
Think step-by-step:
1. What banking products/metrics are mentioned in previous queries?
2. What banking products/metrics are mentioned in current query?
3. Are they the same, related, or completely different?

ANSWER (MUST be one word only):
- If contextually related in banking domain: YES
- If not related or completely different topic: NO

Answer:"""

    def is_query_relevant(
        self,
        current_query: str,
        previous_queries: List[str],
    ) -> Tuple[bool, str]:
        """
        Determine if current query is relevant to previous queries in the thread.
        
        Args:
            current_query: The new user query
            previous_queries: List of previous queries in the thread (most recent first)
            
        Returns:
            Tuple of (is_relevant: bool, reason: str)
        """
        try:
            # If no previous queries, automatically not relevant (start new thread)
            if not previous_queries:
                logger.info("No previous queries, starting new thread")
                return False, "First query in conversation"
            
            # Format previous queries for prompt
            formatted_previous = "\n".join([
                f"{i+1}. {q}" for i, q in enumerate(previous_queries)
            ])
            
            # Generate prompt (simple string formatting)
            prompt = self.relevance_prompt_template.format(
                previous_queries=formatted_previous,
                current_query=current_query,
            )
            
            # Get LLM response
            logger.debug(f"Checking relevance for query: {current_query[:50]}...")
            response = self.llm.call(prompt)
            
            # Parse response
            response_upper = response.upper()
            is_relevant = "YES" in response_upper and "NO" not in response_upper
            
            reason = response if response != "ERROR" else "LLM error, defaulting to new thread"
            
            logger.info(f"Relevance check result: {is_relevant} | Reason: {reason[:100]}")
            return is_relevant, reason
        
        except Exception as e:
            logger.error(f"Error checking query relevance: {e}")
            # On error, default to creating new thread (conservative approach)
            return False, f"Error during relevance check: {str(e)}"

    def check_with_max_history(
        self,
        current_query: str,
        previous_queries: List[str],
        max_history: int = None,
    ) -> Tuple[bool, str]:
        """
        Check relevance with a maximum number of historical queries.
        
        Args:
            current_query: The new user query
            previous_queries: List of previous queries (most recent first)
            max_history: Maximum number of previous queries to consider
            
        Returns:
            Tuple of (is_relevant: bool, reason: str)
        """
        # Use config default if not specified
        if max_history is None:
            max_history = self.conversation_config.get("max_thread_history", 10)
        
        # Limit previous queries to max_history
        limited_queries = previous_queries[:max_history]
        
        return self.is_query_relevant(current_query, limited_queries)


# Singleton instance
_context_checker = None


def get_context_checker() -> ContextChecker:
    """Get or create singleton ContextChecker instance."""
    global _context_checker
    if _context_checker is None:
        _context_checker = ContextChecker()
    return _context_checker


if __name__ == "__main__":
    # Test the context checker
    print("Testing Context Checker...")
    
    checker = get_context_checker()
    print("✓ Context checker initialized")
    
    # Test case 1: Related queries (same topic)
    print("\n--- Test 1: Related Queries ---")
    prev_queries = [
        "What are total deposits in retail banking?",
        "Show me deposits for last quarter",
    ]
    current = "Break down deposits by branch"
    is_relevant, reason = checker.is_query_relevant(current, prev_queries)
    print(f"Current: {current}")
    print(f"Related: {is_relevant}")
    print(f"Reason: {reason}")
    
    # Test case 2: Unrelated queries (different topic)
    print("\n--- Test 2: Unrelated Queries ---")
    prev_queries = [
        "What are total deposits in retail banking?",
        "Show me deposits for last quarter",
    ]
    current = "What are NPA provisions for corporate loans?"
    is_relevant, reason = checker.is_query_relevant(current, prev_queries)
    print(f"Current: {current}")
    print(f"Related: {is_relevant}")
    print(f"Reason: {reason}")
    
    # Test case 3: First query
    print("\n--- Test 3: First Query ---")
    prev_queries = []
    current = "Show me all term loans"
    is_relevant, reason = checker.is_query_relevant(current, prev_queries)
    print(f"Current: {current}")
    print(f"Related: {is_relevant}")
    print(f"Reason: {reason}")
    
    print("\nContext checker is ready!")
