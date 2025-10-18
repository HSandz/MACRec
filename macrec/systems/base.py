import pandas as pd
import streamlit as st
from abc import ABC, abstractmethod
from typing import Any, Optional
from loguru import logger
from langchain.prompts import PromptTemplate

from macrec.agents import Agent
from macrec.utils import is_correct, init_answer, read_json, read_prompts, get_avatar, get_color

class System(ABC):
    """
    The base class of systems. We use the `forward` function to get the system output. Use `set_data` to set the input, context and ground truth answer. Use `is_finished` to check whether the system has finished. Use `is_correct` to check whether the system output is correct. Use `finish` to finish the system and set the system output.
    """
    @staticmethod
    @abstractmethod
    def supported_tasks() -> list[str]:
        """Return a list of supported tasks.

        Raises:
            `NotImplementedError`: Should be implemented in subclasses.
        Returns:
            `list[str]`: A list of supported tasks.
        """
        raise NotImplementedError("System.supported_tasks() not implemented")

    @property
    def task_type(self) -> str:
        """Return the type of the task. Can be inherited by subclasses to support more task types.

        Raises:
            `NotImplementedError`: Not supported task type.
        Returns:
            `str`: The type of the task.
        Example for subclass:
        .. code-block:: python
            class MySystem(System):
                @property
                def task_type(self) -> str:
                    if self.task == 'my_task':
                        return 'my task description'
                    else:
                        return super().task_type
        """
        if self.task == 'qa':
            return 'question answering'
        elif self.task == 'rp':
            return 'rating prediction'
        elif self.task == 'sr':
            return 'ranking'
        elif self.task == 'rr':
            return 'ranking'
        elif self.task == 'chat':
            return 'conversation'
        elif self.task == 'gen':
            return 'explanation generation'
        else:
            # Fallback: use raw task name to avoid crashing in UI flows for new tasks
            return self.task

    def __init__(self, task: str, config_path: str, leak: bool = False, web_demo: bool = False, dataset: Optional[str] = None, *args, **kwargs) -> None:
        """Initialize the system.

        Args:
            `task` (`str`): The task for the system to perform.
            `config_path` (`str`): The path to the config file of the system.
            `leak` (`bool`, optional): Whether to leak the ground truth answer to the system during inference. Defaults to `False`.
            `web_demo` (`bool`, optional): Whether to run the system in web demo mode. Defaults to `False`.
            `dataset` (`str`, optional): The dataset to run in the system. Defaults to `None`.
        """
        self.task = task
        assert self.task in self.supported_tasks()
        self.config = read_json(config_path)
        if 'supported_tasks' in self.config:
            assert isinstance(self.config['supported_tasks'], list) and self.task in self.config['supported_tasks'], f'Task {self.task} is not supported by the system.'
        
        # Handle model override
        self.model_override = kwargs.get('model_override', None)
        self.provider_type = kwargs.get('provider_type', 'openrouter')  # Default to openrouter for backward compatibility
        
        self.agent_kwargs = {
            'system': self,
        }
        if dataset is not None:
            for key, value in self.config.items():
                if isinstance(value, str):
                    self.config[key] = value.format(dataset=dataset, task=self.task)
            self.agent_kwargs['dataset'] = dataset
        self.prompts = read_prompts(self.config['agent_prompt'])
        self.prompts.update(read_prompts(self.config['data_prompt'].format(task=self.task)))
        if 'task_agent_prompt' in self.config:
            self.prompts.update(read_prompts(self.config['task_agent_prompt'].format(task=self.task)))
        self.agent_kwargs['prompts'] = self.prompts
        self.leak = leak
        self.web_demo = web_demo
        self.agent_kwargs['web_demo'] = web_demo
        self.kwargs = kwargs
        self.init(*args, **kwargs)
        self.reset(clear=True)

    def _apply_model_override(self, config: dict) -> dict:
        """Apply model override to a configuration dict."""
        if not self.model_override:
            return config
            
        config = config.copy()
        
        # Set provider type and model name
        config['model_type'] = self.provider_type
        config['model_name'] = self.model_override
        
        if self.provider_type == 'openrouter':
            # Get OpenRouter API key from config
            try:
                api_config = read_json('config/api-config.json')
                if api_config.get('provider') == 'openrouter' and 'api_key' in api_config:
                    config['api_key'] = api_config['api_key']
                    logger.info(f"Using OpenRouter API for model: {self.model_override}")
                else:
                    logger.warning("OpenRouter API key not found in config")
            except Exception as e:
                logger.warning(f"Could not read API config for model override: {e}")
        elif self.provider_type == 'ollama':
            # For Ollama, no API key needed
            logger.info(f"Using Ollama local model: {self.model_override}")
        else:
            logger.warning(f"Unknown provider type: {self.provider_type}")
        
        return config

    def log(self, message: str, agent: Optional[Agent] = None, logging: bool = True) -> None:
        """Log the message.

        Args:
            `message` (`str`): The message to log.
            `agent` (`Agent`, optional): The agent to log the message. Defaults to `None`.
            `logging` (`bool`, optional): Whether to use the `logger` to log the message. Defaults to `True`.
        """
        if logging:
            logger.debug(message)
        if self.web_demo:
            if agent is None:
                role = 'Assistant'
            else:
                role = agent.__class__.__name__
            final_message = f'{get_avatar(role)}:{get_color(role)}[**{role}**]: {message}'
            if 'manager' not in role.lower() and 'assistant' not in role.lower():
                messages = final_message.split('\n')
                messages = [f'- {messages[0]}'] + [f'  {message}' for message in messages[1:]]
                final_message = '\n'.join(messages)
            self.web_log.append(final_message)
            st.markdown(f'{final_message}')

    @abstractmethod
    def init(self, *args, **kwargs) -> None:
        """Initialize the system.

        Raises:
            `NotImplementedError`: Should be implemented in subclasses.
        """
        raise NotImplementedError("System.init() not implemented")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.clear_web_log()
        return self.forward(*args, **kwargs)

    def set_data(self, input: str, context: str, gt_answer: Any, data_sample: Optional[pd.Series] = None) -> None:
        self.input: str = input
        self.context: str = context
        self.gt_answer = gt_answer
        self.data_sample = data_sample
        
        # For SR task, extract candidate_item_id if available
        if self.task == 'sr' and data_sample is not None and 'candidate_item_id' in data_sample:
            candidate_ids = data_sample['candidate_item_id']
            if isinstance(candidate_ids, list):
                # Convert all to string for consistent tracking
                self.kwargs['candidate_items'] = [str(id) for id in candidate_ids]
                logger.debug(f'Set candidate_items for SR task: {len(candidate_ids)} items')

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass of the system.

        Raises:
            `NotImplementedError`: Should be implemented in subclasses.
        Returns:
            `Any`: The system output.
        """
        raise NotImplementedError("System.forward() not implemented")

    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return is_correct(task=self.task, answer=self.answer, gt_answer=self.gt_answer)

    def finish(self, answer: Any) -> str:
        self.answer = answer
        if not self.leak:
            observation = f'The answer you give (may be INCORRECT): {self.answer}'
        elif self.is_correct():
            observation = 'Answer is CORRECT'
        else:
            observation = 'Answer is INCORRECT'
        self.finished = True
        return observation

    def clear_web_log(self) -> None:
        self.web_log = []

    def reset(self, clear: bool = False, *args, **kwargs) -> None:
        self.scratchpad: str = ''
        self.finished: bool = False
        self.answer = init_answer(type=self.task)
        if self.web_demo and clear:
            self.clear_web_log()
