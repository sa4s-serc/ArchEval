"""
SAAM LLM Client - Handles communication with LLM services
Uses GCP Vertex AI with retry logic and rate limit handling matching ATAM implementation.
Supports both Gemini (via google-genai) and Claude (via anthropic-vertex).
"""
import os
import time
import re
from typing import Optional, Tuple, Dict, Any

# Gemini Imports
from google import genai
from google.genai import types, errors as genai_errors

from anthropic import AnthropicVertex, RateLimitError as AnthropicRateLimitError

class LLMClient:
    """
    A specific wrapper for LLM inference handling both Gemini (via google-genai)
    and Claude (via anthropic-vertex) with unified 429 Rate Limit handling.
    """
    
    def __init__(self, 
                 model_name: str = "gemini-3-flash-preview", # or "claude-3-5-sonnet-v2@20241022"
                 project: Optional[str] = None,
                 location: Optional[str] = None):
        """
        Initialize the LLM client with Vertex AI.
        
        Args:
            model_name: Model to use (default: gemini-3-flash-preview or claude-3-5-sonnet-v2@20241022)
            project: GCP project ID (or set GOOGLE_CLOUD_PROJECT env var)
            location: GCP location (or set GOOGLE_CLOUD_LOCATION env var, default: us-central1)
        """
        self.project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "global")
        self.model_name = model_name

        if not self.project:
            raise ValueError("Google Cloud Project ID is required.")

        if "claude" in self.model_name.lower():
            print(f"🔹 Initializing Claude Client ({self.model_name})...")
            self.provider = "anthropic"
            self.client = AnthropicVertex(
                project_id=self.project,
                region=self.location,
            )
        else:
            print(f"🔹 Initializing Gemini Client ({self.model_name})...")
            self.provider = "gemini"
            try:
                self.client = genai.Client(
                    vertexai=True,
                    project=self.project,
                    location=self.location
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Vertex AI Client: {e}")
    
    def query(self, prompt: str, temperature: float = 0.2, max_retries: int = 7, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Send a query to the LLM and return the response with retry logic.
        Compatible with SAAM's existing interface while using Vertex AI backend.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Sampling temperature (default 0.7)
            max_retries: Maximum number of retry attempts for rate limits
            **kwargs: Additional parameters
            
        Returns:
            Tuple[str, Dict[str, Any]]: The LLM response text and usage metadata
        """
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                # --- Gemini Execution ---
                if self.provider == "gemini":
                    # Build config with response_mime_type (default to JSON for structured responses)
                    config_params = {"temperature": temperature}
                    # Default to JSON unless explicitly overridden
                    response_mime_type = kwargs.get("response_mime_type", "application/json")
                    if response_mime_type:
                        config_params["response_mime_type"] = response_mime_type
                    
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(**config_params)
                    )
                    
                    # Try to extract text
                    text = getattr(response, 'text', None)
                    if text is None:
                        try:
                            text = response.text
                        except Exception:
                            text_parts = []
                            if hasattr(response, 'candidates') and response.candidates:
                                for candidate in response.candidates:
                                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                        for part in candidate.content.parts:
                                            if getattr(part, 'type', None) == 'text' and hasattr(part, 'text') and part.text:
                                                text_parts.append(part.text)
                            text = ''.join(text_parts) if text_parts else str(response)

                    # Extract Usage
                    usage = {"input_tokens": 0, "output_tokens": 0, "time_taken": 0}
                    if hasattr(response, 'usage_metadata'):
                        usage["input_tokens"] = response.usage_metadata.prompt_token_count
                        usage["output_tokens"] = response.usage_metadata.candidates_token_count

                # --- Claude Execution ---
                elif self.provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=8192,
                        temperature=temperature,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    
                    # Check for refusal
                    if hasattr(response, 'stop_reason') and response.stop_reason == 'refusal':
                        print(f"\n🚫 CLAUDE REFUSAL DETECTED 🚫")
                        print(f"Claude refused to respond to the prompt.")
                        print(f"Stop reason: {response.stop_reason}")
                        print(f"\n📝 Prompt that caused refusal (first 500 chars):\n{prompt[:500]}...")
                        print(f"\nThis usually means the prompt triggers Claude's content policy.")
                        print(f"Consider rephrasing or simplifying the prompt.\n")
                        text = "{}"
                    else:
                        # Safely extract text from Claude response
                        text = ""
                        if hasattr(response, 'content') and response.content:
                            if len(response.content) > 0:
                                text = response.content[0].text
                            else:
                                print(f"⚠️ Claude returned empty content list")
                                text = "{}"
                        else:
                            print(f"⚠️ Claude response has no content attribute")
                            text = "{}"
                    
                    # Extract Usage
                    usage = {
                        "input_tokens": getattr(response.usage, 'input_tokens', 0) if hasattr(response, 'usage') else 0,
                        "output_tokens": getattr(response.usage, 'output_tokens', 0) if hasattr(response, 'usage') else 0,
                        "time_taken": 0
                    }

                # Common Timing
                end_time = time.time()
                usage["time_taken"] = end_time - start_time
                
                return text, usage

            # --- Error Handling (Unified) ---
            except Exception as e:
                is_rate_limit = False
                
                # Check for Gemini 429
                if isinstance(e, genai_errors.ClientError) and e.code == 429:
                    is_rate_limit = True
                # Check for Claude 429
                elif AnthropicVertex and isinstance(e, AnthropicRateLimitError):
                    is_rate_limit = True

                if is_rate_limit:
                    wait_time = self._extract_wait_time(str(e))
                    if wait_time is None:
                        wait_time = 10 * (2 ** attempt)

                    print(f"⚠️ Quota exceeded (429). Waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ LLM Client Error: {e}")
                    return "{}", {"input_tokens": 0, "output_tokens": 0, "time_taken": 0}
        
        print("❌ Max retries reached. Returning empty response.")
        return "{}", {"input_tokens": 0, "output_tokens": 0, "time_taken": 0}
    
    def _extract_wait_time(self, error_message: str) -> Optional[float]:
        """
        Parse the specific wait time from error messages.
        
        Args:
            error_message: The error message string
            
        Returns:
            float: Wait time in seconds, or None if not found
        """
        try:
            # Search for 'retry in X seconds' or 'retry_after': X
            match = re.search(r"retry in (\d+(\.\d+)?)s", error_message, re.IGNORECASE)
            if match:
                return float(match.group(1)) + 1.0
            
            match_struct = re.search(r"seconds:\s*(\d+)", error_message)
            if match_struct:
                return float(match_struct.group(1)) + 1.0
        except Exception:
            pass
        return None
    
    def generate(self, system_instruction: str, user_content: str, temperature: float = 0.7, max_retries: int = 7, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        ATAM-compatible method: Generates a response using separate system instruction and user content.
        
        Args:
            system_instruction: System-level instructions for the LLM
            user_content: User's actual query/content
            temperature: Sampling temperature
            max_retries: Maximum retry attempts
            **kwargs: Additional parameters (e.g., response_mime_type)
            
        Returns:
            Tuple[str, Dict[str, Any]]: The LLM response text and usage metadata
        """
        full_prompt = f"{system_instruction}\n\nUSER INPUT:\n{user_content}"
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                # --- Gemini Execution ---
                if self.provider == "gemini":
                    # Build config with response_mime_type (default to JSON for structured responses)
                    config_params = {"temperature": temperature}
                    # Default to JSON unless explicitly overridden
                    response_mime_type = kwargs.get("response_mime_type", "application/json")
                    if response_mime_type:
                        config_params["response_mime_type"] = response_mime_type
                    
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=full_prompt,
                        config=types.GenerateContentConfig(**config_params)
                    )
                    text_output = response.text
                    
                    # Extract Usage
                    usage = {"input_tokens": 0, "output_tokens": 0, "time_taken": 0}
                    if hasattr(response, 'usage_metadata'):
                        usage["input_tokens"] = response.usage_metadata.prompt_token_count
                        usage["output_tokens"] = response.usage_metadata.candidates_token_count

                # --- Claude Execution ---
                elif self.provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=8192,  # Claude requires max_tokens to be set
                        temperature=temperature,
                        system=system_instruction,  # Claude supports separate system param
                        messages=[
                            {"role": "user", "content": user_content}
                        ]
                    )
                    
                    # Check for refusal
                    if hasattr(response, 'stop_reason') and response.stop_reason == 'refusal':
                        print(f"\n🚫 CLAUDE REFUSAL DETECTED 🚫")
                        print(f"Claude refused to respond to the prompt.")
                        print(f"Stop reason: {response.stop_reason}")
                        print(f"\n📝 System instruction (first 300 chars):\n{system_instruction[:300]}...")
                        print(f"\n📝 User content (first 500 chars):\n{user_content[:500]}...")
                        print(f"\nThis usually means the prompt triggers Claude's content policy.")
                        print(f"Consider rephrasing or simplifying the prompt.\n")
                        text_output = "{}"
                    else:
                        # Safely extract text from Claude response
                        text_output = ""
                        if hasattr(response, 'content') and response.content:
                            if len(response.content) > 0:
                                text_output = response.content[0].text
                            else:
                                print(f"⚠️ Claude returned empty content list")
                                text_output = "{}"
                        else:
                            print(f"⚠️ Claude response has no content attribute")
                            text_output = "{}"
                    
                    # Extract Usage
                    usage = {
                        "input_tokens": getattr(response.usage, 'input_tokens', 0) if hasattr(response, 'usage') else 0,
                        "output_tokens": getattr(response.usage, 'output_tokens', 0) if hasattr(response, 'usage') else 0,
                        "time_taken": 0
                    }

                # Common Timing
                end_time = time.time()
                usage["time_taken"] = end_time - start_time
                
                return text_output, usage

            # --- Error Handling (Unified) ---
            except Exception as e:
                is_rate_limit = False
                
                # Check for Gemini 429
                if isinstance(e, genai_errors.ClientError) and e.code == 429:
                    is_rate_limit = True
                # Check for Claude 429
                elif AnthropicVertex and isinstance(e, AnthropicRateLimitError):
                    is_rate_limit = True

                if is_rate_limit:
                    wait_time = self._extract_wait_time(str(e))
                    if wait_time is None:
                        wait_time = 10 * (2 ** attempt)

                    print(f"⚠️ Quota exceeded (429). Waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ LLM Client Error: {e}")
                    return "{}", {"input_tokens": 0, "output_tokens": 0, "time_taken": 0}
        
        print("❌ Max retries reached. Returning empty response.")
        return "{}", {"input_tokens": 0, "output_tokens": 0, "time_taken": 0}
    
    def batch_query(self, prompts: list, temperature: float = 0.7, **kwargs) -> list:
        """
        Send multiple queries to the LLM with retry logic for each.
        
        Args:
            prompts: List of prompts to send
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            list: List of response tuples (text, usage)
        """
        responses = []
        for i, prompt in enumerate(prompts):
            print(f"  Processing batch query {i+1}/{len(prompts)}...")
            response = self.query(prompt, temperature=temperature, **kwargs)
            responses.append(response)
        return responses