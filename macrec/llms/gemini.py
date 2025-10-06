from loguru import logger
import google.generativeai as genai
import json
from typing import Any, Dict

from macrec.llms.basellm import BaseLLM

class GeminiLLM(BaseLLM):
    def __init__(self, model_name: str = 'gemini-2.0-flash-001', json_mode: bool = False, agent_context: str = None, *args, **kwargs):
        """Initialize the Gemini LLM.

        Args:
            `model_name` (`str`, optional): The name of the Gemini model. Defaults to `gemini-2.0-flash-001`.
            `json_mode` (`bool`, optional): Whether to use JSON mode. Defaults to `False`.
            `agent_context` (`str`, optional): The context of the agent using this LLM (e.g., 'Manager', 'Analyst'). Defaults to None.
        """
        # Call parent constructor to initialize token tracking attributes
        super().__init__()
        
        self.model_name = model_name
        self.json_mode = json_mode
        self.agent_context = agent_context or "Unknown"
        self.max_tokens: int = kwargs.get('max_tokens', 256)
        
        # Set context length based on model
        if 'gemini-1.5-pro' in model_name:
            self.max_context_length = 2097152  # 2M tokens for Gemini 1.5 Pro
        elif 'gemini-2.0-flash-001' in model_name:
            self.max_context_length = 1048576  # 1M tokens for Gemini 1.5 Flash
        else:
            self.max_context_length = 32768  # Default fallback
        
        # Configure generation parameters
        generation_config = {
            "temperature": kwargs.get('temperature', 0),
            "top_p": kwargs.get('top_p', 0.95),
            "top_k": kwargs.get('top_k', 64),
            "max_output_tokens": self.max_tokens,
        }
        
        if json_mode:
            generation_config["response_mime_type"] = "application/json"
            logger.info("Using JSON mode for Gemini API.")
        
        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config
        )

    @property
    def tokens_limit(self) -> int:
        """Get the token limit for the model."""
        return self.max_context_length
    
    def _make_api_request(self, prompt_text: str):
        """Make a single API request without retry logic.
        
        This wraps the Google SDK call so it can be used with execute_with_retry().
        
        Args:
            prompt_text: The prompt text to send
            
        Returns:
            Response object from Gemini
            
        Raises:
            Exception: For any API errors
        """
        return self.model.generate_content(prompt_text)
    
    def _get_retriable_errors(self) -> tuple:
        """Override to include Google API specific errors.
        
        Returns:
            Tuple of exception types that should trigger retry
        """
        # Get base retriable errors (connection, timeout, etc.)
        base_errors = super()._get_retriable_errors()
        
        # Add Google API specific errors if available
        try:
            from google.api_core import exceptions as google_exceptions
            return base_errors + (
                google_exceptions.ServiceUnavailable,
                google_exceptions.InternalServerError,
                google_exceptions.TooManyRequests,
                google_exceptions.DeadlineExceeded,
            )
        except ImportError:
            # If google.api_core not available, just use base errors
            return base_errors

    def __call__(self, prompt: str, *args, **kwargs) -> str:
        """Forward pass of the Gemini LLM.

        Args:
            `prompt` (`str`): The prompt to feed into the LLM.
        Returns:
            `str`: The Gemini LLM output.
        """
        try:
            # Apply prompt compression if enabled
            final_prompt, compression_info = self.compress_prompt_if_needed(prompt)
            
            # Log the prompt being sent to the API
            logger.info(f"LLM Prompt ({self.agent_context} → {self.model_name}):\n{final_prompt}")
            
            # Log estimated token usage for the prompt
            estimated_prompt_tokens = self.estimate_tokens(final_prompt)
            logger.info(f"📊 Token Usage ({self.agent_context}): ~{estimated_prompt_tokens} prompt tokens estimated")
            
            # Prepare prompt based on JSON mode
            if self.json_mode:
                # For JSON mode, add instruction to the prompt
                actual_prompt = f"{final_prompt}\n\nPlease respond with valid JSON only."
            else:
                actual_prompt = final_prompt
            
            # Make the API request with automatic retry for transient errors
            # Using base class retry mechanism that works for all LLM implementations
            response = self.execute_with_retry(
                self._make_api_request,
                prompt_text=actual_prompt
            )
            
            # Extract text from response
            if response.text:
                content = response.text.replace('\n', ' ').strip()
                
                # Track token usage (Gemini doesn't provide exact counts, so we estimate)
                input_tokens = None
                output_tokens = None
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    input_tokens = getattr(response.usage_metadata, 'prompt_token_count', None)
                    output_tokens = getattr(response.usage_metadata, 'candidates_token_count', None)
                
                # Track the usage including compression info
                self.track_usage(
                    actual_prompt, 
                    content, 
                    input_tokens, 
                    output_tokens,
                    compression_info=compression_info
                )
                
                # Log the response from the API
                logger.info(f"LLM Response ({self.agent_context} → {self.model_name}):\n{content}")
                
                # Log token usage after API response
                if input_tokens and output_tokens:
                    total_tokens = input_tokens + output_tokens
                    logger.info(f"📊 Token Usage ({self.agent_context}): {input_tokens} prompt + {output_tokens} completion = {total_tokens} total tokens")
                else:
                    # Fallback to estimation if no API usage info
                    estimated_input = self.estimate_tokens(actual_prompt)
                    estimated_output = self.estimate_tokens(content)
                    estimated_total = estimated_input + estimated_output
                    logger.info(f"📊 Token Usage ({self.agent_context}): ~{estimated_input} prompt + ~{estimated_output} completion = ~{estimated_total} total tokens (estimated)")
                
                # Log cumulative usage
                logger.info(f"📈 Cumulative Usage ({self.agent_context}): {self.total_input_tokens + (input_tokens or 0)} total tokens across {self.api_calls + 1} calls")
                
                return content
            else:
                logger.warning("Empty response from Gemini API")
                return ""
        
        # Use base class error handling that works for all LLM implementations
        except Exception as e:
            # Use base class error handler for consistent error handling across all LLMs
            # This handles connection errors, timeouts, Google API errors, etc.
            return self.handle_api_error(e)
