from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from loguru import logger

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
