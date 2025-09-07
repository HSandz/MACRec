import json
from jsonformer import Jsonformer
from loguru import logger
from typing import Any
from transformers import pipeline
from transformers.pipelines import Pipeline

from macrec.llms.basellm import BaseLLM

class MyJsonFormer:
    """
    The JsonFormer formatter, which formats the output of the LLM into JSON with the given JSON schema.
    """
    def __init__(self, json_schema: dict, pipeline: Pipeline, max_new_tokens: int = 300, temperature: float = 0.9, debug: bool = False):
        """Initialize the JsonFormer formatter.

        Args:
            `json_schema` (`dict`): The JSON schema of the output.
            `pipeline` (`Pipeline`): The pipeline of the LLM. Must be a `pipeline("text-generation")` pipeline here.
            `max_new_tokens` (`int`, optional): Maximum number of new tokens to generate for each string and number field. Defaults to `300`.
            `temperature` (`float`, optional): The temperature of the generation. Defaults to `0.9`.
            `debug` (`bool`, optional): Whether to enable debug mode. Defaults to `False`.
        """
        self.json_schema = json_schema
        self.pipeline = pipeline
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.debug = debug

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Invoke the JsonFormer formatter.

        Args:
            `prompt` (`str`): The prompt to feed into the LLM.
        Returns:
            `str`: The formatted output. Must be a valid JSON string.
        """
        model = Jsonformer(
            model=self.pipeline.model,
            tokenizer=self.pipeline.tokenizer,
            json_schema=self.json_schema,
            prompt=prompt,
            max_number_tokens=self.max_new_tokens,
            max_string_token_length=self.max_new_tokens,
            debug=self.debug,
            temperature=self.temperature,
        )
        text = model()
        return json.dumps(text, ensure_ascii=False)

class OpenSourceLLM(BaseLLM):
    def __init__(self, model_path: str = 'lmsys/vicuna-7b-v1.5-16k', device: int = 0, json_mode: bool = False, prefix: str = 'react', max_new_tokens: int = 300, do_sample: bool = True, temperature: float = 0.9, top_p: float = 1.0, agent_context: str = None, *args, **kwargs):
        """Initialize the OpenSource LLM. The OpenSource LLM is a wrapper of the HuggingFace pipeline.

        Args:
            `model_path` (`str`, optional): The path or name to the model. Defaults to `'lmsys/vicuna-7b-v1.5-16k'`.
            `device` (`int`, optional): The device to use. Set to `auto` to automatically select the device. Defaults to `0`.
            `json_mode` (`bool`, optional): Whether to enable json mode. If enabled, the output of the LLM will be formatted into JSON by `MyJsonFormer`. Defaults to `False`.
            `prefix` (`str`, optional): The prefix of the some configuration arguments. Defaults to `'react'`.
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
        if device == 'auto':
            self.pipe = pipeline("text-generation", model=model_path, device_map='auto')
        else:
            self.pipe = pipeline("text-generation", model=model_path, device=device)
        self.pipe.model.generation_config.do_sample = do_sample
        self.pipe.model.generation_config.top_p = top_p
        self.pipe.model.generation_config.temperature = temperature
        self.pipe.model.generation_config.max_new_tokens = max_new_tokens
        if self.json_mode:
            logger.info('Enabling json mode...')
            json_schema = kwargs.get(f'{prefix}_json_schema', None)
            assert json_schema is not None, "json_schema must be provided if json_mode is True"
            self.pipe = MyJsonFormer(json_schema=json_schema, pipeline=self.pipe, max_new_tokens=max_new_tokens, temperature=temperature, debug=kwargs.get('debug', False))
        self.max_tokens = max_new_tokens
        self.max_context_length: int = 16384 if '16k' in model_path else 32768 if '32k' in model_path else 4096

    def __call__(self, prompt: str, *args, **kwargs) -> str:
        """Forward pass of the OpenSource LLM. If json_mode is enabled, the output of the LLM will be formatted into JSON by `MyJsonFormer`.

        Args:
            `prompt` (`str`): The prompt to feed into the LLM.
        Returns:
            `str`: The OpenSource LLM output.
        """
        # Apply prompt compression if enabled
        final_prompt, compression_info = self.compress_prompt_if_needed(prompt)
        
        # Log the prompt being sent to the LLM
        logger.info(f"LLM Prompt ({self.agent_context} → {self.model_name}):\n{final_prompt}")
        
        # Log estimated token usage for the prompt
        estimated_prompt_tokens = self.estimate_tokens(final_prompt)
        logger.info(f"📊 Token Usage ({self.agent_context}): ~{estimated_prompt_tokens} prompt tokens estimated")
        
        if self.json_mode:
            result = self.pipe.invoke(final_prompt)
        else:
            result = self.pipe.invoke(final_prompt, return_full_text=False)[0]['generated_text']
        
        # Track usage including compression info
        self.track_usage(
            final_prompt, 
            result, 
            compression_info=compression_info
        )
        
        # Log the response from the LLM
        logger.info(f"LLM Response ({self.agent_context} → {self.model_name}):\n{result}")
        
        # Log token usage after LLM response (estimated for opensource models)
        estimated_input = self.estimate_tokens(final_prompt)
        estimated_output = self.estimate_tokens(result)
        estimated_total = estimated_input + estimated_output
        logger.info(f"📊 Token Usage ({self.agent_context}): ~{estimated_input} prompt + ~{estimated_output} completion = ~{estimated_total} total tokens (estimated)")
        
        # Log cumulative usage
        logger.info(f"📈 Cumulative Usage ({self.agent_context}): {self.total_input_tokens + estimated_input} total tokens across {self.api_calls + 1} calls")
        
        return result
