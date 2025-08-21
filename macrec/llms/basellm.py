from abc import ABC, abstractmethod
from typing import Dict, Any

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

    def track_usage(self, prompt: str, response: str, input_tokens: int = None, output_tokens: int = None) -> None:
        """Track token usage for this LLM call.
        
        Args:
            `prompt` (`str`): Input prompt.
            `response` (`str`): LLM response.
            `input_tokens` (`int`, optional): Actual input token count if available.
            `output_tokens` (`int`, optional): Actual output token count if available.
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
            'response_length': len(response)
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
