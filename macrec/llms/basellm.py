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
            `**kwargs`: Additional tracking info like compression_info.
        """
        if input_tokens is None:
            input_tokens = self.estimate_tokens(prompt)
        if output_tokens is None:
            output_tokens = self.estimate_tokens(response)
            
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.api_calls += 1
        
        self.call_history.append({
            'call_id': self.api_calls,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'model': self.model_name,
            'prompt_length': len(prompt),
            'response_length': len(response),
            'compression_info': kwargs.get('compression_info', {})
        })

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
            logger.info(f"Enabled prompt compression for {self.model_name}")
        else:
            logger.info(f"Disabled prompt compression for {self.model_name}")

    def _simple_compress(self, text: str, ratio: float) -> str:
        """Simple rule-based text compression to avoid API calls during compression.
        
        Args:
            text: Text to compress
            ratio: Target compression ratio (0.1-0.9)
            
        Returns:
            Compressed text
        """
        import re
        
        # Target length
        target_length = int(len(text) * ratio)
        
        if len(text) <= target_length:
            return text
        
        # Apply compression rules in order of importance
        compressed = text
        
        # 1. Remove extra whitespace
        compressed = re.sub(r'\s+', ' ', compressed)
        compressed = compressed.strip()
        
        # 2. Remove redundant phrases if still too long
        if len(compressed) > target_length:
            # Remove common filler words while preserving structure
            fillers = [
                r'\b(very|really|quite|rather|extremely|incredibly|absolutely)\s+',
                r'\b(please|kindly)\s+',
                r'\b(I think that|I believe that|It seems that)\s+',
                r'\b(in order to|so as to)\b',
                r'\b(due to the fact that|because of the fact that)\b'
            ]
            
            for filler in fillers:
                compressed = re.sub(filler, '', compressed, flags=re.IGNORECASE)
                if len(compressed) <= target_length:
                    break
        
        # 3. If still too long, truncate sentences intelligently
        if len(compressed) > target_length:
            sentences = re.split(r'[.!?]+', compressed)
            result = ""
            
            for sentence in sentences:
                if len(result + sentence) <= target_length:
                    result += sentence + ". "
                else:
                    break
            
            compressed = result.strip()
        
        # 4. Final truncate if necessary (preserve end of text for context)
        if len(compressed) > target_length:
            # Keep the first part and last part
            first_part = compressed[:target_length//2]
            last_part = compressed[-(target_length//2):]
            compressed = first_part + "..." + last_part
        
        return compressed

    def compress_prompt_if_needed(self, prompt: str) -> tuple[str, Dict[str, Any]]:
        """Compress prompt if compression is enabled and prompt is long enough.
        
        Args:
            `prompt` (`str`): Original prompt.
            
        Returns:
            `tuple[str, Dict[str, Any]]`: (final_prompt, compression_info)
        """
        compression_info = {
            'compressed': False,
            'original_length': len(prompt),
            'compressed_length': len(prompt),
            'token_savings': 0
        }
        
        if not self.enable_compression:
            return prompt, compression_info
        
        try:
            # Use simple rule-based compression instead of LLM-based compression
            # to avoid recursion and API calls during compression
            try:
                # Simple compression: remove extra whitespace, redundant words, etc.
                compressed_prompt = self._simple_compress(prompt, self.compression_ratio)
                
                original_length = len(prompt)
                compressed_length = len(compressed_prompt)
                actual_ratio = compressed_length / original_length if original_length > 0 else 1.0
                
                # Estimate token savings (rough approximation: 1 token ≈ 4 characters)
                original_tokens = original_length // 4
                compressed_tokens = compressed_length // 4
                token_savings = original_tokens - compressed_tokens
                
                compression_info.update({
                    'compressed': True,
                    'compressed_length': compressed_length,
                    'token_savings': token_savings,
                    'compression_ratio': actual_ratio,
                    'from_cache': False
                })
                
                logger.debug(f"Simple compressed prompt for {self.model_name}: "
                            f"{original_length} -> {compressed_length} chars "
                            f"(~{token_savings} tokens saved)")
                
                return compressed_prompt, compression_info
                original_length = len(prompt)
                compressed_length = len(compressed_prompt)
                actual_ratio = compressed_length / original_length if original_length > 0 else 1.0
                token_savings = (original_length - compressed_length) // 4  # Rough estimation
                
                compression_info.update({
                    'compressed': True,
                    'compressed_length': compressed_length,
                    'token_savings': token_savings,
                    'compression_ratio': actual_ratio,
                    'from_cache': False
                })
                
                logger.debug(f"Compressed prompt for {self.model_name}: "
                            f"{original_length} -> {compressed_length} chars "
                            f"(~{token_savings} tokens saved)")
                
                return compressed_prompt, compression_info
                
            except ImportError:
                logger.debug("LLMLingua not available, skipping compression")
                return prompt, compression_info
                
        except Exception as e:
            logger.warning(f"Failed to compress prompt for {self.model_name}: {e}")
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
