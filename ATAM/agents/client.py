import os
from pydoc import text
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
                 model_name: str = "gemini-2.0-flash-exp",
                 project: Optional[str] = None,
                 location: Optional[str] = None):
        
        self.project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
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

    def generate(self, system_instruction: str, user_content: str, temperature: float = 0.3, max_retries: int = 7, max_tokens: int = 8192) -> Tuple[str, Dict[str, Any]]:
        full_prompt = f"{system_instruction}\n\nUSER INPUT:\n{user_content}"
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                # --- Gemini Execution ---
                if self.provider == "gemini":
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=full_prompt, # Gemini prefers fused prompts or specific parts construction
                        config=types.GenerateContentConfig(
                            temperature=temperature,
                            response_mime_type="application/json"
                        )
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
                        max_tokens=max_tokens,  # Claude requires max_tokens to be set
                        timeout=1200,
                        temperature=temperature,
                        system=system_instruction, # Claude supports separate system param
                        messages=[
                            {"role": "user", "content": user_content}
                        ]
                    )
                    
                    # Robust text extraction for Claude
                    if hasattr(response, 'content') and response.content:
                        text_output = response.content[0].text
                    else:
                        stop_reason = getattr(response, 'stop_reason', 'unknown')
                        print(f"  ⚠️ Claude response content empty. Stop reason: {stop_reason}")
                        text_output = f"{{'error': 'Empty response', 'stop_reason': '{stop_reason}'}}"
                    
                    # Extract Usage
                    usage = {
                        "input_tokens": getattr(response.usage, 'input_tokens', 0),
                        "output_tokens": getattr(response.usage, 'output_tokens', 0),
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

    def _extract_wait_time(self, error_message: str) -> Optional[float]:
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