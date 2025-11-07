"""
Token Counter Utility

Provides functionality to count tokens using vLLM's chat completions endpoint
with dry_run mode for accurate token counting.
"""

import requests
import logging
from typing import Dict, Optional

logger = logging.getLogger("ai_insight.token_counter")


class TokenCounter:
    """Token counter using vLLM API dry_run mode."""
    
    def __init__(self, api_url: str, model_name: str, timeout: int = 10):
        """
        Initialize token counter.
        
        Args:
            api_url: Base API URL (e.g., "http://10.16.1.102:8000/v1")
            model_name: Model name for token counting
            timeout: Request timeout in seconds
        """
        self.api_url = api_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
    
    def count_tokens(self, text: str) -> Dict[str, int]:
        """
        Count tokens using the chat completions endpoint with dry_run=True.
        If dry_run is not supported, falls back to making a minimal request
        and extracting token counts from the usage field.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            dict: Token counts with keys 'prompt_tokens', 'completion_tokens', 'total_tokens'
                  Returns error dict if request fails
        """
        url = f"{self.api_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        
        # Try with dry_run first
        data_dry_run = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 1,
            "dry_run": True
        }
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json=data_dry_run,
                timeout=self.timeout
            )
            
            # If dry_run is supported, use it
            if response.status_code == 200:
                result = response.json()
                if 'usage' in result:
                    token_data = {
                        'prompt_tokens': result['usage'].get('prompt_tokens', 0),
                        'completion_tokens': result['usage'].get('completion_tokens', 0),
                        'total_tokens': result['usage'].get('total_tokens', 0)
                    }
                    logger.debug(f"Token count (dry_run): {token_data['total_tokens']} tokens")
                    return token_data
            
        except Exception as e:
            logger.debug(f"dry_run attempt failed: {e}")
        
        # If dry_run not supported (404 or error), try regular call with minimal tokens
        logger.debug("dry_run not supported, using minimal generation for token count")
        
        try:
            data_minimal = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": text}],
                "max_tokens": 1,  # Generate only 1 token
                "temperature": 0.0
            }
            
            response = requests.post(
                url,
                headers=headers,
                json=data_minimal,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract token counts from the response
            if 'usage' in result:
                token_data = {
                    'prompt_tokens': result['usage'].get('prompt_tokens', 0),
                    'completion_tokens': result['usage'].get('completion_tokens', 0),
                    'total_tokens': result['usage'].get('total_tokens', 0)
                }
                logger.debug(f"Token count (minimal gen): {token_data['total_tokens']} tokens")
                return token_data
            
            logger.warning("Token count not found in API response")
            return {"error": "Token count not found in response", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP Error: {e.response.status_code} - {e.response.text}"
            logger.error(f"Token counting failed: {error_msg}")
            return {"error": error_msg, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(f"Token counting failed: {error_msg}")
            return {"error": error_msg, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def create_token_counter(config: dict) -> Optional[TokenCounter]:
    """
    Create a TokenCounter from configuration.
    
    Args:
        config: Configuration dict with vLLM settings
        
    Returns:
        TokenCounter instance or None if config missing
    """
    vllm_cfg = config.get("vllm", {})
    api_url = vllm_cfg.get("api_url")
    model_name = vllm_cfg.get("model_name")
    
    if not api_url or not model_name:
        logger.warning("vLLM configuration missing for token counter")
        return None
    
    return TokenCounter(api_url, model_name)
