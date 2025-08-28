from loguru import logger
import google.generativeai as genai
import json
from typing import Any, Dict

from macrec.llms.basellm import BaseLLM

class GeminiLLM(BaseLLM):
    def __init__(self, model_name: str = 'gemini-2.0-flash-001', json_mode: bool = False, *args, **kwargs):
        """Initialize the Gemini LLM.

        Args:
            `model_name` (`str`, optional): The name of the Gemini model. Defaults to `gemini-2.0-flash-001`.
            `json_mode` (`bool`, optional): Whether to use JSON mode. Defaults to `False`.
        """
        # Call parent constructor to initialize token tracking attributes
        super().__init__()
        
        self.model_name = model_name
        self.json_mode = json_mode
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
            
            if self.json_mode:
                # For JSON mode, add instruction to the prompt
                json_prompt = f"{final_prompt}\n\nPlease respond with valid JSON only."
                response = self.model.generate_content(json_prompt)
                actual_prompt = json_prompt
            else:
                response = self.model.generate_content(final_prompt)
                actual_prompt = final_prompt
            
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
                
                return content
            else:
                logger.warning("Empty response from Gemini API")
                return ""
                
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return f"Error: {str(e)}"
