from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from loguru import logger
import time
import requests

class BaseLLM(ABC):
    def __init__(self) -> None:
        self.model_name: str
        self.max_tokens: int
        self.max_context_length: int
        self.json_mode: bool
        # Token tracking
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.api_calls: int = 0
        self.call_history: list = []
        # Prompt compression settings
        self.enable_compression: bool = False
        self.compression_ratio: float = 0.5
        self.prompt_compressor: Optional[Any] = None
        # Error tracking for compression
        self.compression_errors: int = 0
        self.max_compression_errors: int = 3
        # Retry configuration
        self.max_retries: int = 3
        self.retry_delay_base: int = 1  # Base delay in seconds for exponential backoff

    @property
    def tokens_limit(self) -> int:
        """Limit of tokens that can be fed into the LLM under the current context length.

        Returns:
            `int`: The limit of tokens that can be fed into the LLM under the current context length.
        """
        return self.max_context_length - 2 * self.max_tokens - 50  # single round need 2 agent prompt steps: thought and action

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text. Override in subclasses for better accuracy.
        
        Args:
            `text` (`str`): Text to estimate tokens for.
        Returns:
            `int`: Estimated token count.
        """
        # Simple estimation: ~4 characters per token
        return len(text) // 4

    def track_usage(self, prompt: str, response: str, input_tokens: int = None, output_tokens: int = None, **kwargs) -> None:
        """Track token usage for this LLM call.
        
        Args:
            `prompt` (`str`): Input prompt.
            `response` (`str`): LLM response.
            `input_tokens` (`int`, optional): Actual input token count if available.
            `output_tokens` (`int`, optional): Actual output token count if available.
            `**kwargs`: Additional tracking info like compression_info, api_usage.
        """
        # Use actual API token counts when available, otherwise estimate
        api_usage = kwargs.get('api_usage', {})
        estimated_input = False
        estimated_output = False
        
        if input_tokens is None:
            input_tokens = self.estimate_tokens(prompt)
            estimated_input = True
            
        if output_tokens is None:
            output_tokens = self.estimate_tokens(response)
            estimated_output = True
            
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.api_calls += 1
        
        # Store detailed call information
        call_info = {
            'call_id': self.api_calls,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'model': self.model_name,
            'prompt_length': len(prompt),
            'response_length': len(response),
            'compression_info': kwargs.get('compression_info', {}),
            'estimated_input': estimated_input,
            'estimated_output': estimated_output,
            'api_usage': api_usage  # Store full API usage info
        }
        
        self.call_history.append(call_info)
        
        # Log if we're using estimates vs actual API counts
        if estimated_input or estimated_output:
            logger.debug(f"Token tracking for call {self.api_calls}: estimated_input={estimated_input}, estimated_output={estimated_output}")
        else:
            logger.debug(f"Token tracking for call {self.api_calls}: using actual API token counts")

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this LLM.
        
        Returns:
            `Dict[str, Any]`: Usage statistics.
        """
        return {
            'model_name': self.model_name,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'api_calls': self.api_calls,
            'avg_input_tokens': self.total_input_tokens / max(self.api_calls, 1),
            'avg_output_tokens': self.total_output_tokens / max(self.api_calls, 1),
        }
    
    def get_detailed_usage_stats(self) -> Dict[str, Any]:
        """Get detailed usage statistics including API vs estimated token breakdown.
        
        Returns:
            `Dict[str, Any]`: Detailed usage statistics.
        """
        api_calls_with_actual = sum(1 for call in self.call_history 
                                  if not call.get('estimated_input', True) and not call.get('estimated_output', True))
        api_calls_with_estimates = self.api_calls - api_calls_with_actual
        
        return {
            'model_name': self.model_name,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'api_calls': self.api_calls,
            'api_calls_with_actual_counts': api_calls_with_actual,
            'api_calls_with_estimated_counts': api_calls_with_estimates,
            'avg_input_tokens': self.total_input_tokens / max(self.api_calls, 1),
            'avg_output_tokens': self.total_output_tokens / max(self.api_calls, 1),
            'accuracy_rate': api_calls_with_actual / max(self.api_calls, 1),
        }

    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.api_calls = 0
        self.call_history = []

    def enable_prompt_compression(
        self, 
        enable: bool = True, 
        compression_ratio: float = 0.5
    ) -> None:
        """Enable or disable prompt compression.
        
        Args:
            `enable` (`bool`): Whether to enable compression.
            `compression_ratio` (`float`): Compression ratio (0.1-0.9).
        """
        self.enable_compression = enable
        self.compression_ratio = compression_ratio
        
        if enable:
            # Initialize API-based prompt compressor
            try:
                from macrec.utils.prompt_compression import APIPromptCompressor
                
                # Create a separate LLM instance for compression to avoid recursion
                compression_llm = self._create_compression_llm()
                
                if compression_llm is None:
                    logger.warning("Could not create compression LLM, disabling compression")
                    self.enable_compression = False
                    return
                
                self.prompt_compressor = APIPromptCompressor(
                    compression_ratio=compression_ratio,
                    cache_dir="cache/prompts",
                    enable_cache=True,
                    min_compression_length=200,
                    preserve_structure=True,
                    llm_instance=compression_llm
                )
                
                logger.info(f"Enabled API-based prompt compression for {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize API-based compression: {e}")
                self.enable_compression = False
        else:
            self.prompt_compressor = None
            logger.info(f"Disabled prompt compression for {self.model_name}")
    
    def _create_compression_llm(self):
        """Create a lightweight LLM instance for compression tasks.
        
        This should be implemented by subclasses to avoid recursion.
        """
        try:
            # Import here to avoid circular imports
            from macrec.llms.openrouter import OpenRouterLLM
            from macrec.llms.ollama import OllamaLLM
            
            # Try OpenRouter models first
            compression_models = [
                ('openrouter', 'google/gemini-2.0-flash-lite-001'),  # Free and fast
                ('openrouter', 'openai/gpt-3.5-turbo'),  # Reliable fallback
                ('openrouter', 'anthropic/claude-3-haiku'),  # Another reliable option
            ]
            
            # Add Ollama models as fallbacks
            try:
                # Check if Ollama is available
                ollama_test = OllamaLLM(model_name='llama3.2')
                if ollama_test._check_ollama_server():
                    ollama_models = ollama_test.list_models()
                    # Add common lightweight models if available
                    for model in ['llama3.2:1b', 'llama3.2', 'gemma:2b', 'phi3']:
                        if any(model in available for available in ollama_models):
                            compression_models.append(('ollama', model))
            except:
                pass  # Ollama not available, skip
            
            for provider, model in compression_models:
                try:
                    if provider == 'openrouter':
                        compression_llm = OpenRouterLLM(
                            model_name=model,
                            max_tokens=512,  # Keep compression responses short
                            temperature=0.3  # Lower temperature for consistent compression
                        )
                    elif provider == 'ollama':
                        compression_llm = OllamaLLM(
                            model_name=model,
                            max_tokens=512,
                            temperature=0.3
                        )
                    else:
                        continue
                    
                    # IMPORTANT: Disable compression for the compression LLM to prevent recursion
                    compression_llm.enable_compression = False
                    compression_llm.compression_ratio = 1.0
                    
                    # Test the compression LLM with a simple prompt
                    test_response = compression_llm("Test compression. Respond with 'OK'.")
                    if test_response and "error" not in test_response.lower():
                        logger.debug(f"Successfully created compression LLM with {provider} model: {model}")
                        return compression_llm
                    else:
                        logger.warning(f"Compression LLM test failed for {provider}/{model}: {test_response}")
                        
                except Exception as e:
                    logger.debug(f"Failed to create compression LLM with {provider}/{model}: {e}")
                    continue
            
            # If all models fail, return None
            logger.warning("Could not create any compression LLM")
            return None
            
        except Exception as e:
            logger.error(f"Error creating compression LLM: {e}")
            return None

    def compress_prompt_if_needed(self, prompt: str) -> tuple[str, Dict[str, Any]]:
        """Compress prompt using API-based compression if enabled.
        
        Args:
            `prompt` (`str`): Original prompt.
            
        Returns:
            `tuple[str, Dict[str, Any]]`: (final_prompt, compression_info)
        """
        compression_info = {
            'compressed': False,
            'original_length': len(prompt),
            'compressed_length': len(prompt),
            'token_savings': 0,
            'compression_ratio': 1.0,
            'from_cache': False
        }
        
        if not self.enable_compression or not self.prompt_compressor:
            return prompt, compression_info
        
        try:
            # Use API-based compression
            result = self.prompt_compressor.compress_prompt(
                prompt=prompt,
                compression_ratio=self.compression_ratio
            )
            
            # Update compression info with results
            compression_info.update({
                'compressed': result.get('compression_enabled', False),
                'compressed_length': result.get('compressed_length', len(prompt)),
                'token_savings': result.get('token_savings', 0),
                'compression_ratio': result.get('compression_ratio_actual', 1.0),
                'from_cache': result.get('from_cache', False)
            })
            
            compressed_prompt = result.get('compressed_prompt', prompt)
            
            if result.get('compression_enabled', False):
                logger.debug(f"API compressed prompt for {self.model_name}: "
                            f"{result['original_length']} -> {result['compressed_length']} chars "
                            f"(ratio: {result['compression_ratio_actual']:.2f}, "
                            f"~{result['token_savings']} tokens saved, "
                            f"cached: {result['from_cache']})")
            
            return compressed_prompt, compression_info
                
        except Exception as e:
            # Track compression errors and disable if too many occur
            self.compression_errors += 1
            logger.warning(f"Failed to compress prompt for {self.model_name}: {e} (error #{self.compression_errors})")
            
            if self.compression_errors >= self.max_compression_errors:
                logger.error(f"Too many compression errors ({self.compression_errors}), disabling compression for {self.model_name}")
                self.enable_compression = False
                self.prompt_compressor = None
            
            return prompt, compression_info

    def execute_with_retry(
        self,
        api_call_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute an API call with automatic retry for transient network errors.
        
        This method provides a generic retry mechanism that can be used by all LLM implementations.
        It handles connection errors, timeouts, and HTTP server errors with exponential backoff.
        
        Args:
            api_call_func: The function to call (e.g., requests.post, grpc call, etc.)
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the API call
            
        Raises:
            Exception: After all retries are exhausted, raises the last exception
        """
        # Define retriable error types (can be extended by subclasses)
        RETRIABLE_ERRORS = self._get_retriable_errors()
        
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                # Execute the API call
                result = api_call_func(*args, **kwargs)
                
                # Check if result indicates a server error that should be retried
                if self._should_retry_response(result):
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay_base * (2 ** attempt)
                        logger.warning(
                            f"🔄 Server error in response (attempt {attempt + 1}/{self.max_retries}). "
                            f"Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        continue
                
                # Success
                return result
                
            except tuple(RETRIABLE_ERRORS) as e:
                last_exception = e
                error_type = type(e).__name__
                
                if attempt < self.max_retries - 1:
                    # Calculate exponential backoff delay: 1s, 2s, 4s
                    wait_time = self.retry_delay_base * (2 ** attempt)
                    
                    logger.warning(
                        f"🔄 Transient error (attempt {attempt + 1}/{self.max_retries}): "
                        f"{error_type}: {str(e)[:100]}... Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    # Final attempt failed
                    logger.error(
                        f"❌ Error persisted after {self.max_retries} attempts: "
                        f"{error_type}: {str(e)[:200]}"
                    )
                    raise
            
            except Exception as e:
                # Non-retriable error - check if it should be retried
                if self._is_retriable_exception(e):
                    last_exception = e
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay_base * (2 ** attempt)
                        logger.warning(
                            f"🔄 Retriable error (attempt {attempt + 1}/{self.max_retries}): "
                            f"{type(e).__name__}: {str(e)[:100]}... Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        continue
                
                # Non-retriable error
                logger.error(f"❌ Non-retriable error: {type(e).__name__}: {str(e)[:200]}")
                raise
        
        # This should not be reached, but just in case
        if last_exception:
            raise last_exception
        
        return None
    
    def _get_retriable_errors(self) -> tuple:
        """Get tuple of error types that should trigger retry.
        
        Can be overridden by subclasses to add custom retriable errors.
        
        Returns:
            Tuple of exception classes that should be retried
        """
        try:
            return (
                requests.exceptions.ConnectionError,  # Includes ConnectionResetError, ConnectionAbortedError
                requests.exceptions.Timeout,
                ConnectionResetError,
                ConnectionAbortedError,
                BrokenPipeError,
                TimeoutError,
                OSError,  # Network unreachable, etc.
            )
        except Exception:
            # If requests is not available (e.g., for non-HTTP LLMs)
            return (
                ConnectionResetError,
                ConnectionAbortedError,
                BrokenPipeError,
                TimeoutError,
                OSError,
            )
    
    def _should_retry_response(self, response: Any) -> bool:
        """Check if a response indicates a retriable error.
        
        Can be overridden by subclasses to handle provider-specific response checks.
        
        Args:
            response: The API response object
            
        Returns:
            True if the response indicates a retriable error, False otherwise
        """
        # Default implementation for HTTP responses
        if hasattr(response, 'status_code'):
            # Retry on server errors (5xx) and rate limiting (429)
            return response.status_code >= 500 or response.status_code == 429
        
        return False
    
    def _is_retriable_exception(self, exception: Exception) -> bool:
        """Check if an exception should trigger a retry.
        
        Can be overridden by subclasses to add custom retry logic.
        
        Args:
            exception: The exception that was raised
            
        Returns:
            True if should retry, False otherwise
        """
        # Check for HTTP errors that should be retried
        if hasattr(exception, 'response') and hasattr(exception.response, 'status_code'):
            status_code = exception.response.status_code
            # Retry on server errors (5xx) and rate limiting (429)
            return status_code >= 500 or status_code == 429
        
        return False
    
    def classify_error(self, exception: Exception) -> tuple[str, str]:
        """Classify an error for better error messages and handling.
        
        Args:
            exception: The exception to classify
            
        Returns:
            Tuple of (error_code, error_message)
        """
        error_type = type(exception).__name__
        error_str = str(exception)
        
        # Connection errors (transient)
        if isinstance(exception, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return ("CONNECTION_ERROR", f"Network connection lost. {error_type}: {error_str[:100]}")
        
        # Timeout errors
        if isinstance(exception, (TimeoutError,)):
            return ("TIMEOUT", f"Request exceeded time limit. {error_type}")
        
        # Try to handle requests library errors if available
        try:
            import requests
            
            if isinstance(exception, requests.exceptions.ConnectionError):
                return ("CONNECTION_ERROR", f"Network connection error. {error_type}: {error_str[:100]}")
            
            if isinstance(exception, requests.exceptions.Timeout):
                return ("TIMEOUT", "Request timed out after retries")
            
            if isinstance(exception, requests.exceptions.HTTPError):
                status_code = exception.response.status_code if hasattr(exception, 'response') else 'unknown'
                return ("HTTP_ERROR", f"HTTP_{status_code} - Server returned error")
            
            if isinstance(exception, requests.exceptions.RequestException):
                return ("NETWORK_ERROR", f"Request error. {error_type}: {error_str[:100]}")
                
        except ImportError:
            pass
        
        # JSON decode errors
        try:
            import json
            if isinstance(exception, json.JSONDecodeError):
                return ("INVALID_JSON", "Failed to parse API response")
        except ImportError:
            pass
        
        # Generic errors
        return ("UNEXPECTED", f"{error_type}: {error_str[:100]}")
    
    def handle_api_error(self, exception: Exception) -> str:
        """Handle an API error and return a user-friendly error message.
        
        Args:
            exception: The exception that was raised
            
        Returns:
            Error message string to return to the caller
        """
        error_code, error_message = self.classify_error(exception)
        logger.error(f"❌ {error_code}: {error_message}")
        return f"Error: {error_code} - {error_message}"

    @abstractmethod
    def __call__(self, prompt: str, *args, **kwargs) -> str:
        """Forward pass of the LLM.

        Args:
            `prompt` (`str`): The prompt to feed into the LLM.
        Raises:
            `NotImplementedError`: Should be implemented in subclasses.
        Returns:
            `str`: The LLM output.
        """
        raise NotImplementedError("BaseLLM.__call__() not implemented")
