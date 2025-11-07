"""
Integration of Dynamic Prompt Builder with Existing SQL Generation

This module integrates the new dynamic prompt builder into the existing
sql_generator.py workflow.
"""

import sys
import os
from typing import Tuple, List, Dict, Optional
import logging

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from Scripts.dynamic_prompt_builder import DynamicPromptBuilder, build_dynamic_prompt_string, get_dynamic_prompt_builder

logger = logging.getLogger("ai_insight.dynamic_prompt_integration")


class DynamicPromptOrchestrator:
    """Orchestrate dynamic prompt generation with existing components."""
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.builder = get_dynamic_prompt_builder()
    
    def generate_dynamic_prompt(
        self, 
        user_query: str,
        retrieved_examples: Optional[List[Dict]] = None,
        use_examples: bool = True
    ) -> Tuple[str, Dict]:
        """
        Generate a dynamic prompt for the given query.
        
        Args:
            user_query: Natural language query
            retrieved_examples: Optional list of similar examples from vector DB
            use_examples: Whether to include examples in prompt
            
        Returns:
            Tuple of (prompt_string, metadata_dict)
        """
        try:
            examples = retrieved_examples if use_examples else None
            prompt_string, metadata = build_dynamic_prompt_string(user_query, examples)
            
            logger.info(f"Generated dynamic prompt with {metadata['num_rules']} rules")
            return prompt_string, metadata
            
        except Exception as e:
            logger.error(f"Failed to generate dynamic prompt: {e}")
            raise
    
    def compare_approaches(
        self,
        user_query: str,
        static_prompt: str,
        retrieved_examples: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Compare dynamic prompt approach vs static prompt approach.
        
        Returns:
            Dict with comparison metrics
        """
        dynamic_prompt, dynamic_metadata = self.generate_dynamic_prompt(
            user_query, 
            retrieved_examples
        )
        
        comparison = {
            "query": user_query,
            "static_prompt_size": len(static_prompt),
            "dynamic_prompt_size": len(dynamic_prompt),
            "size_reduction": len(static_prompt) - len(dynamic_prompt),
            "reduction_percent": round(
                (len(static_prompt) - len(dynamic_prompt)) / len(static_prompt) * 100,
                2
            ),
            "token_savings_estimate": int((len(static_prompt) - len(dynamic_prompt)) / 4),
            "keywords_extracted": dynamic_metadata.get('num_keywords', 0),
            "rules_applied": dynamic_metadata.get('num_rules', 0),
            "sections_used": dynamic_metadata.get('sections', []),
        }
        
        return comparison


def integrate_with_generate_sql(
    user_query: str,
    use_dynamic_prompt: bool = True,
    retrieved_examples: Optional[List[Dict]] = None
) -> Tuple[str, Dict]:
    """
    Drop-in replacement/wrapper for prompt generation in generate_sql.
    
    This can replace the call to generate_sql_with_metadata() in sql_generator.py
    
    Args:
        user_query: User's natural language query
        use_dynamic_prompt: Whether to use dynamic prompting
        retrieved_examples: Examples from vector database
        
    Returns:
        Tuple of (prompt_string, metadata)
    """
    if not use_dynamic_prompt:
        # Fallback to existing vectorized prompt
        from prompt_orchestrator import generate_sql_with_metadata
        return generate_sql_with_metadata(user_query)
    
    # Use dynamic prompt builder
    orchestrator = DynamicPromptOrchestrator()
    return orchestrator.generate_dynamic_prompt(user_query, retrieved_examples)


# ============================================================================
# Configuration for Dynamic Prompting
# ============================================================================

DYNAMIC_PROMPT_CONFIG = {
    "enabled": True,  # Enable/disable dynamic prompting globally
    "use_examples": True,  # Include examples in dynamic prompt
    "min_keywords": 1,  # Minimum keywords to extract
    "fallback_to_static": False,  # Fallback to static prompt if dynamic fails
    "semantic_matching": False,  # Use semantic similarity for keyword matching (requires sklearn)
    "log_comparison_metrics": True,  # Log efficiency metrics
}


def should_use_dynamic_prompt(query: str) -> bool:
    """Determine if dynamic prompting should be used for a query."""
    if not DYNAMIC_PROMPT_CONFIG["enabled"]:
        return False
    
    # Could add heuristics here
    # e.g., don't use dynamic for very short queries, etc.
    return len(query.split()) >= DYNAMIC_PROMPT_CONFIG["min_keywords"]


# ============================================================================
# Example Integration Code (commented out, for reference)
# ============================================================================

"""
# In sql_generator.py, replace this:

    # Generate the vectorized prompt with metadata
    vectorized_prompt, metadata = generate_sql_with_metadata(query)

# With this:

    # Generate prompt (using dynamic builder if enabled)
    from dynamic_prompt_integration import integrate_with_generate_sql, should_use_dynamic_prompt
    
    use_dynamic = should_use_dynamic_prompt(query)
    if use_dynamic:
        vectorized_prompt, metadata = integrate_with_generate_sql(
            query, 
            use_dynamic_prompt=True,
            retrieved_examples=None  # Can pass examples here
        )
    else:
        vectorized_prompt, metadata = generate_sql_with_metadata(query)
"""
