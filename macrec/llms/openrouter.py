import requests
import json
from loguru import logger
from typing import Any, Dict

from macrec.llms.basellm import BaseLLM

class OpenRouterLLM(BaseLLM):
    def __init__(self, model_name: str = 'mistralai/mistral-7b-instruct', api_key: str = '', json_mode: bool = False, *args, **kwargs):
        """Initialize the OpenRouter LLM.

        Args:
            `model_name` (`str`, optional): The name of the model on OpenRouter. Defaults to `mistralai/mistral-7b-instruct`.
            `api_key` (`str`): The API key for OpenRouter. If empty, will try to get from environment or config.
            `json_mode` (`bool`, optional): Whether to use JSON mode. Defaults to `False`.
        """
        # Call parent constructor to initialize token tracking attributes
        super().__init__()
        
        self.model_name = model_name
        self.json_mode = json_mode
        self.max_tokens: int = kwargs.get('max_tokens', 1024)
        self.temperature: float = kwargs.get('temperature', 0.7)
        self.top_p: float = kwargs.get('top_p', 0.95)
        
        # Set up API key - try multiple sources
        if api_key:
            self.api_key = api_key
        else:
            # Try to get from environment variables or config
            import os
            self.api_key = os.getenv('OPENROUTER_API_KEY', '')
            if not self.api_key:
                try:
                    from macrec.utils import read_json
                    api_config = read_json('config/api-config.json')
                    if api_config.get('provider') == 'openrouter':
                        self.api_key = api_config.get('api_key', '')
                    elif api_config.get('provider') == 'mixed':
                        self.api_key = api_config.get('openrouter_api_key', '')
                except:
                    pass
        
        if not self.api_key:
            logger.warning("OpenRouter API key not found. Please set it in config/api-config.json or as OPENROUTER_API_KEY environment variable.")
        
        # Set context length based on model
        # Default context lengths for common models
        model_context_lengths = {
            'mistralai/mistral-7b-instruct': 32768,
            'meta-llama/llama-3.1-8b-instruct': 131072,
            'anthropic/claude-3-haiku': 200000,
            'openai/gpt-3.5-turbo': 16384,
            'openai/gpt-4': 8192,
            'openai/gpt-4-turbo': 128000,
            'openai/gpt-oss-20b:free': 8192,
            'google/gemini-pro': 32768,
            'google/gemini-2.0-flash-001': 1048576,
            'google/gemini-2.0-flash-lite-001': 1048576,
            'google/gemini-1.5-pro': 2097152,
            'google/gemini-1.5-flash': 1048576,
            'cohere/command-r': 128000,
            'microsoft/wizardlm-2-8x22b': 65536,
        }
        
        self.max_context_length = model_context_lengths.get(model_name, 32768)  # Default fallback
        
        # Set up headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Base URL for OpenRouter
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        logger.info(f"Initialized OpenRouter LLM with model: {model_name}")

    @property
    def tokens_limit(self) -> int:
        """Get the token limit for the model."""
        return self.max_context_length

    def __call__(self, prompt: str, *args, **kwargs) -> str:
        """Forward pass of the OpenRouter LLM.

        Args:
            `prompt` (`str`): The prompt to feed into the LLM.
        Returns:
            `str`: The OpenRouter LLM output.
        """
        try:
            # Apply prompt compression if enabled
            final_prompt, compression_info = self.compress_prompt_if_needed(prompt)
            
            # Prepare the request payload
            messages = [{"role": "user", "content": final_prompt}]
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            
            # Add JSON mode if enabled
            if self.json_mode:
                payload["response_format"] = {"type": "json_object"}
                # Add instruction to the prompt for JSON mode
                messages[0]["content"] = f"{final_prompt}\n\nPlease respond with valid JSON only."
            
            # Make the API request
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=120  # 2 minute timeout
            )
            
            # Check if request was successful
            if response.status_code == 200:
                result = response.json()
                
                # Extract the response text and usage information
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    
                    # Track token usage if available in response
                    input_tokens = None
                    output_tokens = None
                    if 'usage' in result:
                        input_tokens = result['usage'].get('prompt_tokens')
                        output_tokens = result['usage'].get('completion_tokens')
                    
                    # Track the usage including compression info
                    self.track_usage(
                        final_prompt, 
                        content, 
                        input_tokens, 
                        output_tokens,
                        compression_info=compression_info
                    )
                    
                    return content.strip()
                else:
                    logger.warning("No choices found in OpenRouter API response")
                    return ""
            else:
                logger.error(f"OpenRouter API request failed with status {response.status_code}: {response.text}")
                return f"Error: HTTP {response.status_code}"
                
        except requests.exceptions.Timeout:
            logger.error("OpenRouter API request timed out")
            return "Error: Request timed out"
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to OpenRouter API: {e}")
            return f"Error: {str(e)}"
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing OpenRouter API response: {e}")
            return f"Error: Invalid JSON response"
        except Exception as e:
            logger.error(f"Unexpected error calling OpenRouter API: {e}")
            return f"Error: {str(e)}"
