"""
Prompt Logging Module

Logs prompt generation details including timestamp, query, retrieved examples,
and token counts to a CSV file for analysis and monitoring.
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger("ai_insight.prompt_logger")
logger.setLevel(logging.ERROR)


@dataclass
class PromptLogEntry:
    """Data class for a single prompt log entry."""
    timestamp: str
    user_query: str
    example_1_id: str
    example_1_snippet: str  # Shortened preview
    example_2_id: str
    example_2_snippet: str  # Shortened preview
    prompt_token_count: int
    completion_token_count: int
    total_token_count: int
    error: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV writing."""
        return asdict(self)


class PromptLogger:
    """Logger for prompt generation details."""
    
    # CSV Headers
    HEADERS = [
        "timestamp",
        "user_query",
        "example_1_id",
        "example_1_snippet",
        "example_2_id",
        "example_2_snippet",
        "prompt_token_count",
        "completion_token_count",
        "total_token_count",
        "error"
    ]
    
    def __init__(self, log_dir: str = "logs", log_filename: str = "prompt_generation.csv"):
        """
        Initialize prompt logger.
        
        Args:
            log_dir: Directory to store log files
            log_filename: Name of the CSV log file
        """
        self.log_dir = Path(log_dir)
        self.log_file = self.log_dir / log_filename
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Create CSV file with headers if it doesn't exist
        if not self.log_file.exists():
            self._create_log_file()
    
    def _create_log_file(self):
        """Create CSV file with headers."""
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADERS)
            writer.writeheader()
    
    def log_prompt_generation(
        self,
        user_query: str,
        retrieved_examples: List[Dict],
        token_counts: Dict[str, int],
        error: str = ""
    ):
        """
        Log a prompt generation event.
        
        Args:
            user_query: The user's natural language query
            retrieved_examples: List of retrieved example dicts with 'id' and 'text' keys
            token_counts: Dict with 'prompt_tokens', 'completion_tokens', 'total_tokens'
            error: Optional error message
        """
        timestamp = datetime.now().isoformat()
        
        # Extract examples (handle cases with fewer than 2 examples)
        example_1_id = retrieved_examples[0].get('id', '') if len(retrieved_examples) > 0 else ''
        example_1_text = retrieved_examples[0].get('text', '') if len(retrieved_examples) > 0 else ''
        example_2_id = retrieved_examples[1].get('id', '') if len(retrieved_examples) > 1 else ''
        example_2_text = retrieved_examples[1].get('text', '') if len(retrieved_examples) > 1 else ''
        
        # Create readable snippets (extract just the User query line from examples)
        def extract_snippet(example_text: str, max_len: int = 100) -> str:
            """Extract a readable snippet from the example text."""
            if not example_text:
                return ""
            # Try to extract the "User:" line
            lines = example_text.split('\n')
            for line in lines:
                if line.strip().startswith('User:'):
                    snippet = line.strip()[5:].strip()  # Remove "User:" prefix
                    return snippet[:max_len] + "..." if len(snippet) > max_len else snippet
            # Fallback to first line
            return example_text[:max_len] + "..." if len(example_text) > max_len else example_text
        
        example_1_snippet = extract_snippet(example_1_text, 150)
        example_2_snippet = extract_snippet(example_2_text, 150)
        
        entry = PromptLogEntry(
            timestamp=timestamp,
            user_query=user_query,
            example_1_id=example_1_id,
            example_1_snippet=example_1_snippet,
            example_2_id=example_2_id,
            example_2_snippet=example_2_snippet,
            prompt_token_count=token_counts.get('prompt_tokens', 0),
            completion_token_count=token_counts.get('completion_tokens', 0),
            total_token_count=token_counts.get('total_tokens', 0),
            error=error
        )
        
        try:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.HEADERS)
                writer.writerow(entry.to_dict())
            
            # Suppress non-error logging
        except Exception as e:
            logger.error(f"Failed to write prompt log entry: {e}")


# Global logger instance (lazy initialization)
_prompt_logger: Optional[PromptLogger] = None


def get_prompt_logger(log_dir: str = "logs", log_filename: str = "prompt_generation.csv") -> PromptLogger:
    """
    Get or create the global prompt logger instance.
    
    Args:
        log_dir: Directory to store log files
        log_filename: Name of the CSV log file
        
    Returns:
        PromptLogger instance
    """
    global _prompt_logger
    if _prompt_logger is None:
        _prompt_logger = PromptLogger(log_dir, log_filename)
    return _prompt_logger
