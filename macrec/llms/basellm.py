from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from loguru import logger
import time
import requests

class BaseLLM(ABC):
    def __init__(self) -> None:
        self.model: str
        self.max_tokens: int
        self.max_context_length: int
        self.json_mode: bool
        # Token tracking
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.api_calls: int = 0
        self.call_history: list = []
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
            `**kwargs`: Additional tracking info like api_usage.
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
            'model': self.model,
            'prompt_length': len(prompt),
            'response_length': len(response),
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
            'model': self.model,
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
            'model': self.model,
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
