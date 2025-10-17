import requests
import json
from loguru import logger
from typing import Any, Dict

from macrec.llms.basellm import BaseLLM

class OllamaLLM(BaseLLM):
    def __init__(self, model_name: str = 'llama3.2:1b', base_url: str = 'http://localhost:11434', json_mode: bool = False, agent_context: str = None, *args, **kwargs):
        """Initialize the Ollama LLM.

        Args:
            `model_name` (`str`, optional): The name of the model in Ollama. Defaults to `llama3.2`.
            `base_url` (`str`, optional): The base URL for the Ollama API. Defaults to `http://localhost:11434`.
            `json_mode` (`bool`, optional): Whether to use JSON mode. Defaults to `False`.
            `agent_context` (`str`, optional): The context of the agent using this LLM (e.g., 'Manager', 'Analyst'). Defaults to None.
        """
        # Call parent constructor to initialize token tracking attributes
        super().__init__()
        
        self.model_name = model_name
        self.json_mode = json_mode
        self.agent_context = agent_context or "Unknown"
        self.max_tokens: int = kwargs.get('max_tokens', 1024)
        self.temperature: float = kwargs.get('temperature', 0.7)
        self.top_p: float = kwargs.get('top_p', 0.9)
        self.top_k: int = kwargs.get('top_k', 40)
        self.timeout: int = kwargs.get('timeout', 300)  # 5 minutes for local models
        
        # Set up Ollama configuration
        # Try to get base_url from config file if not explicitly provided
        if base_url == 'http://localhost:11434':  # default value
            try:
                from macrec.utils import read_json
                api_config = read_json('config/api-config.json')
                if api_config.get('ollama_base_url'):
                    base_url = api_config.get('ollama_base_url')
            except:
                pass  # Use default if config not found
        
        self.base_url = base_url.rstrip('/')
        self.generate_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
        
        # Set context length based on model
        # Default context lengths for common Ollama models
        model_context_lengths = {
            'llama3.2': 131072,
            'llama3.2:1b': 131072,
            'llama3.2:3b': 131072,
            'llama3.1': 131072,
            'llama3.1:8b': 131072,
            'llama3.1:70b': 131072,
            'llama3': 8192,
            'llama2': 4096,
            'llama2:7b': 4096,
            'llama2:13b': 4096,
            'llama2:70b': 4096,
            'codellama': 16384,
            'codellama:7b': 16384,
            'codellama:13b': 16384,
            'codellama:34b': 16384,
            'mistral': 32768,
            'mistral:7b': 32768,
            'mixtral': 32768,
            'mixtral:8x7b': 32768,
            'gemma': 8192,
            'gemma:2b': 8192,
            'gemma:7b': 8192,
            'phi': 2048,
            'phi3': 4096,
            'qwen': 32768,
            'qwen2': 32768,
            'neural-chat': 4096,
            'starling-lm': 8192,
            'yi': 4096,
            'dolphin-mixtral': 32768,
            'orca-mini': 4096,
            'vicuna': 2048,
            'wizard-vicuna': 2048,
        }
        
        self.max_context_length = model_context_lengths.get(model_name, 4096)  # Default fallback
        
        # Set up headers for API requests
        self.headers = {
            "Content-Type": "application/json"
        }
        
        logger.info(f"Initialized Ollama LLM with model: {model_name} at {base_url}")

    @property
    def tokens_limit(self) -> int:
        """Get the token limit for the model."""
        return self.max_context_length
    
    def _make_api_request(
        self,
        payload: Dict[str, Any],
        timeout: int = 300
    ) -> requests.Response:
        """Make a single API request without retry logic.
        
        This is the core request method that will be wrapped by execute_with_retry().
        
        Args:
            payload: The request payload
            timeout: Request timeout in seconds (default 5 minutes for local models)
            
        Returns:
            Response object
            
        Raises:
            requests.exceptions.RequestException: For any request errors
        """
        return requests.post(
            self.generate_url,
            headers=self.headers,
            json=payload,
            timeout=timeout
        )

    def _check_ollama_server(self) -> bool:
        """Check if Ollama server is running and the model is available."""
        try:
            # Check if server is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            
            # Check if the model is available
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            # Check exact match or partial match for model name
            if self.model_name in model_names:
                return True
            
            # Check for partial matches (e.g., 'llama3.2' matches 'llama3.2:latest')
            for model_name in model_names:
                if self.model_name in model_name or model_name.startswith(self.model_name):
                    logger.info(f"Found model match: {model_name} for requested {self.model_name}")
                    return True
            
            logger.warning(f"Model {self.model_name} not found in Ollama. Available models: {model_names}")
            logger.info(f"You can pull the model by running: ollama pull {self.model_name}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect to Ollama server at {self.base_url}: {e}")
            logger.info("Make sure Ollama is installed and running. Visit https://ollama.ai for installation instructions.")
            return False

    def __call__(self, prompt: str, *args, **kwargs) -> str:
        """Forward pass of the Ollama LLM.

        Args:
            `prompt` (`str`): The prompt to feed into the LLM.
        Returns:
            `str`: The Ollama LLM output.
        """
        try:
            # Check if Ollama server is available
            if not self._check_ollama_server():
                error_msg = f"Ollama server not available or model {self.model_name} not found"
                logger.error(error_msg)
                return f"Error: {error_msg}"
            
            # Apply prompt compression if enabled
            final_prompt, compression_info = self.compress_prompt_if_needed(prompt)
            
            # Log the prompt being sent to the API
            logger.info(f"LLM Prompt ({self.agent_context} → {self.model_name}):\n{final_prompt}")
            
            # Log estimated token usage for the prompt
            estimated_prompt_tokens = self.estimate_tokens(final_prompt)
            logger.info(f"📊 Token Usage ({self.agent_context}): ~{estimated_prompt_tokens} prompt tokens estimated")
            
            # Prepare the request payload for Ollama generate API
            payload = {
                "model": self.model_name,
                "prompt": final_prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "num_predict": self.max_tokens
                }
            }
            
            # Add JSON mode if enabled
            if self.json_mode:
                payload["format"] = "json"
                # Add explicit JSON instruction to prompt
                final_prompt = f"{final_prompt}\n\nPlease respond with valid JSON only."
                payload["prompt"] = final_prompt
            
            # Make the API request with automatic retry for transient errors
            # Using base class retry mechanism that works for all LLM implementations
            response = self.execute_with_retry(
                self._make_api_request,
                payload=payload,
                timeout=self.timeout  # Use configurable timeout
            )
            
            # Log response summary for debugging
            logger.debug(f"Ollama API response: status={response.status_code}, length={len(response.text)} chars")
            
            # Check response size - if too large, it might be malformed
            if len(response.text) > 500000:  # 500KB limit
                logger.warning(f"Very large response from Ollama: {len(response.text)} chars")
            
            # Check if request was successful
            if response.status_code == 200:
                # Check content type before parsing JSON
                content_type = response.headers.get('content-type', '')
                if 'application/json' not in content_type:
                    logger.warning(f"Unexpected content-type from Ollama: {content_type}")
                
                # Enhanced JSON parsing with better error handling
                try:
                    result = response.json()
                except json.JSONDecodeError as json_err:
                    # Log the response details for debugging
                    response_text = response.text
                    logger.error(f"JSON decode error in Ollama response:")
                    logger.error(f"  Error: {json_err}")
                    logger.error(f"  Response length: {len(response_text)} chars")
                    logger.error(f"  Response preview (first 500 chars): {response_text[:500]}")
                    logger.error(f"  Response preview (last 500 chars): {response_text[-500:]}")
                    
                    # Try to extract content manually if possible
                    import re
                    content_match = re.search(r'"response":\s*"([^"]*)"', response_text)
                    if content_match:
                        logger.warning("Attempting to extract content manually from malformed JSON")
                        content = content_match.group(1)
                        
                        # Track usage with estimates since we can't parse the JSON
                        self.track_usage(
                            final_prompt, 
                            content, 
                            None,  # Will use estimation
                            None,  # Will use estimation
                            compression_info=compression_info
                        )
                        
                        logger.info(f"LLM Response ({self.agent_context} → {self.model_name}):\n{content}")
                        return content
                    
                    return f"Error: INVALID_JSON - Failed to parse Ollama response"
                
                # Extract the response text
                if 'response' in result:
                    content = result['response'].strip()
                    
                    # Extract token usage info if available
                    input_tokens = None
                    output_tokens = None
                    
                    if 'prompt_eval_count' in result:
                        input_tokens = result['prompt_eval_count']
                    if 'eval_count' in result:
                        output_tokens = result['eval_count']
                    
                    # Track usage including compression info
                    self.track_usage(
                        final_prompt, 
                        content, 
                        input_tokens, 
                        output_tokens,
                        compression_info=compression_info,
                        api_usage=result
                    )
                    
                    # Log the response from the API
                    logger.info(f"LLM Response ({self.agent_context} → {self.model_name}):\n{content}")
                    
                    # Log token usage after API response
                    if input_tokens and output_tokens:
                        total_tokens = input_tokens + output_tokens
                        logger.info(f"📊 Token Usage ({self.agent_context}): {input_tokens} prompt + {output_tokens} completion = {total_tokens} total tokens")
                    else:
                        # Fallback to estimation if no API usage info
                        estimated_input = self.estimate_tokens(final_prompt)
                        estimated_output = self.estimate_tokens(content)
                        estimated_total = estimated_input + estimated_output
                        logger.info(f"📊 Token Usage ({self.agent_context}): ~{estimated_input} prompt + ~{estimated_output} completion = ~{estimated_total} total tokens (estimated)")
                    
                    # Log cumulative usage
                    logger.info(f"� Cumulative Usage ({self.agent_context}): {self.total_input_tokens + self.total_output_tokens} total tokens across {self.api_calls} calls")
                    
                    return content
                else:
                    logger.error(f"No 'response' field in Ollama response: {result}")
                    return "Error: Invalid response format from Ollama"
                    
            else:
                # Handle HTTP errors using base class method for consistency
                error_msg = f"Ollama API request failed with status {response.status_code}"
                try:
                    error_detail = response.json()
                    if 'error' in error_detail:
                        error_msg += f": {error_detail['error']}"
                except:
                    error_msg += f": {response.text[:200]}"
                
                logger.error(error_msg)
                # Use base class error handler for consistent error classification
                return self.handle_api_error(Exception(error_msg))
        
        # Use base class error handling that works for all LLM implementations
        except json.JSONDecodeError as e:
            # JSON errors are specific to response parsing, not the base request
            logger.error(f"❌ JSON decode error: {e}")
            return f"Error: INVALID_JSON - Failed to parse API response"
        
        except Exception as e:
            # Use base class error handler for consistent error handling across all LLMs
            # This handles connection errors, timeouts, etc.
            return self.handle_api_error(e)

    def list_models(self) -> list:
        """List available models in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            else:
                logger.error(f"Failed to list Ollama models: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error listing Ollama models: {e}")
            return []

    def pull_model(self, model_name: str = None) -> bool:
        """Pull a model in Ollama."""
        if model_name is None:
            model_name = self.model_name
            
        try:
            payload = {"name": model_name}
            response = requests.post(
                f"{self.base_url}/api/pull",
                headers=self.headers,
                json=payload,
                timeout=1800  # 30 minute timeout for model pulling
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully pulled model {model_name}")
                return True
            else:
                logger.error(f"Failed to pull model {model_name}: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
