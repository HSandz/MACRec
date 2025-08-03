from loguru import logger
from transformers import AutoTokenizer
from langchain.prompts import PromptTemplate

from macrec.agents.base import Agent
from macrec.llms import GeminiLLM
from macrec.utils import format_step, run_once

class Manager(Agent):
    """
    The manager agent. The manager agent is a two-stage agent, which first prompts the thought LLM and then prompts the action LLM.
    """
    def __init__(self, thought_config_path: str, action_config_path: str, *args, **kwargs) -> None:
        """Initialize the manager agent. The manager agent is a two-stage agent, which first prompts the thought LLM and then prompts the action LLM.

        Args:
            `thought_config_path` (`str`): The path to the config file of the thought LLM.
            `action_config_path` (`str`): The path to the config file of the action LLM.
        """
        super().__init__(*args, **kwargs)
        self.thought_llm = self.get_LLM(thought_config_path)
        self.action_llm = self.get_LLM(action_config_path)
        self.json_mode = self.action_llm.json_mode
        
        # Initialize tokenizers based on LLM type
        if isinstance(self.thought_llm, GeminiLLM):
            # For Gemini, we'll use a simple word-based estimation
            self.thought_enc = None
        else:
            self.thought_enc = AutoTokenizer.from_pretrained(self.thought_llm.model_name)
            
        if isinstance(self.action_llm, GeminiLLM):
            # For Gemini, we'll use a simple word-based estimation
            self.action_enc = None
        else:
            self.action_enc = AutoTokenizer.from_pretrained(self.action_llm.model_name)

    def over_limit(self, **kwargs) -> bool:
        prompt = self._build_manager_prompt(**kwargs)
        
        # For Gemini models, use a simple word-based estimation (4 chars per token approximately)
        if self.action_enc is None:
            action_tokens = len(prompt) // 4
        else:
            action_tokens = len(self.action_enc.encode(prompt))
            
        if self.thought_enc is None:
            thought_tokens = len(prompt) // 4
        else:
            thought_tokens = len(self.thought_enc.encode(prompt))
            
        return action_tokens > self.action_llm.tokens_limit or thought_tokens > self.thought_llm.tokens_limit

    @property
    def manager_prompt(self) -> PromptTemplate:
        if self.json_mode:
            return self.prompts['manager_prompt_json']
        else:
            return self.prompts['manager_prompt']

    @property
    def valid_action_example(self) -> str:
        if self.json_mode:
            return self.prompts['valid_action_example_json'].replace('{finish}', self.prompts['finish_json'])
        else:
            return self.prompts['valid_action_example'].replace('{finish}', self.prompts['finish'])

    @property
    def fewshot_examples(self) -> str:
        if self.json_mode:
            if 'fewshot_examples_json' in self.prompts:
                return self.prompts['fewshot_examples_json']
            elif 'fewshot_examples' in self.prompts:
                return self.prompts['fewshot_examples']
        else:
            if 'fewshot_examples' in self.prompts:
                return self.prompts['fewshot_examples']
        return ''

    @property
    def hint(self) -> str:
        if 'hint' in self.prompts:
            return self.prompts['hint']
        else:
            return ''

    @run_once
    def _log_prompt(self, prompt: str) -> None:
        logger.debug(f'Manager Prompt: {prompt}')

    def _build_manager_prompt(self, **kwargs) -> str:
        return self.manager_prompt.format(
            examples=self.fewshot_examples,
            **kwargs
        )

    def _prompt_thought(self, **kwargs) -> str:
        thought_prompt = self._build_manager_prompt(**kwargs)
        self._log_prompt(thought_prompt)
        thought_response = self.thought_llm(thought_prompt)
        return format_step(thought_response)

    def _prompt_action(self, **kwargs) -> str:
        action_prompt = self._build_manager_prompt(**kwargs)
        action_response = self.action_llm(action_prompt)
        return format_step(action_response)

    def forward(self, stage: str, *args, **kwargs) -> str:
        if stage == 'thought':
            return self._prompt_thought(**kwargs)
        elif stage == 'action':
            return self._prompt_action(**kwargs)
        else:
            raise ValueError(f"Unsupported stage: {stage}")
