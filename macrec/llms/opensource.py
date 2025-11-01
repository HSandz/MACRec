import json
from loguru import logger
from typing import Any, Dict, Optional
from transformers import pipeline
from transformers.pipelines import Pipeline

from macrec.llms.basellm import BaseLLM

class OpenSourceLLM(BaseLLM):
    def __init__(self, model_path: str = 'lmsys/vicuna-7b-v1.5-16k', device: int = 0, json_mode: bool = False, max_new_tokens: int = 300, do_sample: bool = True, temperature: float = 0.9, top_p: float = 1.0, agent_context: str = None, *args, **kwargs):
        """Initialize the OpenSource LLM. The OpenSource LLM is a wrapper of the HuggingFace pipeline.

        Args:
            `model_path` (`str`, optional): The path or name to the model. Defaults to `'lmsys/vicuna-7b-v1.5-16k'`.
            `device` (`int`, optional): The device to use. Set to `'auto'` to automatically select the device. Defaults to `0`.
            `json_mode` (`bool`, optional): Whether to enable json mode. Defaults to `False`.
            `max_new_tokens` (`int`, optional): Maximum number of new tokens to generate. Defaults to `300`.
            `do_sample` (`bool`, optional): Whether to use sampling. Defaults to `True`.
            `temperature` (`float`, optional): The temperature of the generation. Defaults to `0.9`.
            `top_p` (`float`, optional): The top-p of the generation. Defaults to `1.0`.
            `agent_context` (`str`, optional): The context of the agent using this LLM (e.g., 'Manager', 'Analyst'). Defaults to None.
        """
        # Call parent constructor to initialize token tracking attributes
        super().__init__()
        
        self.model_name = model_path
        self.json_mode = json_mode
        self.agent_context = agent_context or "Unknown"
        self.max_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        
        # Initialize the pipeline
        try:
            if device == 'auto':
                self.pipe = pipeline("text-generation", model=model_path, device_map='auto')
            else:
                # Check if device is available, fall back to CPU if not
                import torch
                if isinstance(device, int) and device >= 0:
                    if not torch.cuda.is_available() or device >= torch.cuda.device_count():
                        logger.warning(f"GPU device {device} not available, falling back to CPU")
                        self.pipe = pipeline("text-generation", model=model_path, device='cpu')
                    else:
                        self.pipe = pipeline("text-generation", model=model_path, device=device)
                else:
                    self.pipe = pipeline("text-generation", model=model_path, device=device)
                
            # Configure generation parameters
            if hasattr(self.pipe.model, 'generation_config'):
                self.pipe.model.generation_config.do_sample = do_sample
                self.pipe.model.generation_config.top_p = top_p
                self.pipe.model.generation_config.temperature = temperature
                self.pipe.model.generation_config.max_new_tokens = max_new_tokens
                
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace pipeline for {model_path}: {e}")
            raise RuntimeError(f"Could not load model {model_path}: {e}")
        
        # Set context length based on model name
        self.max_context_length: int = 16384 if '16k' in model_path else 32768 if '32k' in model_path else 4096
        
        # Override retry configuration if provided
        self.max_retries = kwargs.get('max_retries', 3)
        self.retry_delay_base = kwargs.get('retry_delay_base', 1)
        
        logger.info(f"Initialized OpenSource LLM with model: {model_path}, device: {device}, json_mode: {json_mode}")

    @property
    def tokens_limit(self) -> int:
        """Get the token limit for the model."""
        return self.max_context_length

    def _get_retriable_errors(self) -> tuple:
        """Override to include HuggingFace specific errors.
        
        Returns:
            Tuple of exception types that should trigger retry
        """
        # Get base retriable errors (connection, timeout, etc.)
        base_errors = super()._get_retriable_errors()
        
        # Add HuggingFace specific errors if available
        try:
            from transformers.utils import TensorFlowNotFoundError, TorchNotFoundError
            from torch.cuda import OutOfMemoryError as CudaOutOfMemoryError
            return base_errors + (
                TensorFlowNotFoundError,
                TorchNotFoundError,
                CudaOutOfMemoryError,
                RuntimeError,  # General PyTorch/CUDA errors
                MemoryError,   # System memory errors
            )
        except ImportError:
            # If specific libraries not available, just use base errors + common runtime errors
            return base_errors + (
                RuntimeError,
                MemoryError,
            )

    def _make_pipeline_request(self, prompt: str) -> Dict[str, Any]:
        """Make a single pipeline request without retry logic.
        
        This is the core request method that will be wrapped by execute_with_retry().
        
        Args:
            prompt: The prompt text to send to the pipeline
            
        Returns:
            Pipeline output
            
        Raises:
            Exception: For any pipeline errors
        """
        # Add JSON instruction if JSON mode is enabled
        if self.json_mode:
            final_prompt = f"{prompt}\n\nPlease respond with valid JSON only."
        else:
            final_prompt = prompt
        
        # Generate response using the pipeline
        # Use eos_token_id if available, otherwise use None
        eos_token_id = None
        if hasattr(self.pipe, 'tokenizer') and hasattr(self.pipe.tokenizer, 'eos_token_id'):
            eos_token_id = self.pipe.tokenizer.eos_token_id
        
        result = self.pipe(
            final_prompt,
            max_new_tokens=self.max_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            return_full_text=False,
            pad_token_id=eos_token_id
        )
        
        return result

    def __call__(self, prompt: str, *args, **kwargs) -> str:
        """Forward pass of the OpenSource LLM.

        Args:
            `prompt` (`str`): The prompt to feed into the LLM.
        Returns:
            `str`: The OpenSource LLM output.
        """
        try:
            # Apply prompt compression if enabled
            final_prompt, compression_info = self.compress_prompt_if_needed(prompt)
            
            # Log the prompt being sent to the LLM
            logger.info(f"LLM Prompt ({self.agent_context} → {self.model_name}):\n{final_prompt}")
            
            # Log estimated token usage for the prompt
            estimated_prompt_tokens = self.estimate_tokens(final_prompt)
            logger.info(f"📊 Token Usage ({self.agent_context}): ~{estimated_prompt_tokens} prompt tokens estimated")
            
            # Make the pipeline request with automatic retry for transient errors
            # Using base class retry mechanism that works for all LLM implementations
            result = self.execute_with_retry(
                self._make_pipeline_request,
                prompt=final_prompt
            )
            
            # Extract text from pipeline result
            if result and len(result) > 0 and 'generated_text' in result[0]:
                content = result[0]['generated_text'].strip()
                
                # Track usage (local models don't provide exact token counts, so we estimate)
                input_tokens = None  # Will be estimated in track_usage
                output_tokens = None  # Will be estimated in track_usage
                
                # Track the usage including compression info
                self.track_usage(
                    final_prompt, 
                    content, 
                    input_tokens, 
                    output_tokens,
                    compression_info=compression_info
                )
                
                # Log the response from the LLM
                logger.info(f"LLM Response ({self.agent_context} → {self.model_name}):\n{content}")
                
                # Log token usage after LLM response (estimated for opensource models)
                estimated_input = self.estimate_tokens(final_prompt)
                estimated_output = self.estimate_tokens(content)
                estimated_total = estimated_input + estimated_output
                logger.info(f"📊 Token Usage ({self.agent_context}): ~{estimated_input} prompt + ~{estimated_output} completion = ~{estimated_total} total tokens (estimated)")
                
                # Log cumulative usage
                logger.info(f"📈 Cumulative Usage ({self.agent_context}): {self.total_input_tokens} total tokens across {self.api_calls} calls")
                
                return content
            else:
                logger.warning("Empty or invalid response from HuggingFace pipeline")
                return ""
                
        # Use base class error handling that works for all LLM implementations
        except Exception as e:
            # Use base class error handler for consistent error handling across all LLMs
            # This handles CUDA errors, memory errors, model loading errors, etc.
            return self.handle_api_error(e)

    def forward(self, prompt: str, *args, **kwargs) -> str:
        """Alternative forward method for compatibility.
        
        Args:
            `prompt` (`str`): The prompt to feed into the LLM.
        Returns:
            `str`: The OpenSource LLM output.
        """
        return self.__call__(prompt, *args, **kwargs)
