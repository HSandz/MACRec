from loguru import logger
import google.generativeai as genai
import json
from typing import Any, Dict

from macrec.llms.basellm import BaseLLM

class GeminiLLM(BaseLLM):
    def __init__(self, model_name: str = 'gemini-2.0-flash', json_mode: bool = False, *args, **kwargs):
        """Initialize the Gemini LLM.

        Args:
            `model_name` (`str`, optional): The name of the Gemini model. Defaults to `gemini-2.0-flash`.
            `json_mode` (`bool`, optional): Whether to use JSON mode. Defaults to `False`.
        """
        self.model_name = model_name
        self.json_mode = json_mode
        self.max_tokens: int = kwargs.get('max_tokens', 256)
        
        # Set context length based on model
        if 'gemini-1.5-pro' in model_name:
            self.max_context_length = 2097152  # 2M tokens for Gemini 1.5 Pro
        elif 'gemini-2.0-flash' in model_name:
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
            if self.json_mode:
                # For JSON mode, add instruction to the prompt
                json_prompt = f"{prompt}\n\nPlease respond with valid JSON only."
                response = self.model.generate_content(json_prompt)
            else:
                response = self.model.generate_content(prompt)
            
            # Extract text from response
            if response.text:
                return response.text.replace('\n', ' ').strip()
            else:
                logger.warning("Empty response from Gemini API")
                return ""
                
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return f"Error: {str(e)}"
