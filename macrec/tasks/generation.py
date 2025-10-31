import os
import pandas as pd
from abc import abstractmethod
from tqdm import tqdm
from typing import Any
from loguru import logger
from argparse import ArgumentParser
import datetime

from macrec.tasks.base import Task
from macrec.utils import init_api, read_json, token_tracker, duration_tracker
from macrec.systems import CollaborationSystem, ReWOOSystem

class GenerationTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--api_config', type=str, default='config/api-config.json', help='Api configuration file')
        parser.add_argument('--dataset', type=str, default='None', help='Dataset name')
        parser.add_argument('--data_file', type=str, required=True, help='Dataset file')
        parser.add_argument('--system', type=str, default='react', choices=['react', 'reflection', 'analyse', 'collaboration', 'rewoo'], help='System name')
        parser.add_argument('--system_config', type=str, required=True, help='System configuration file')
        parser.add_argument('--model', type=str, default='google/gemini-2.0-flash-001', help='Model name for all agents')
        parser.add_argument('--task', type=str, default='rp', choices=['rp', 'sr', 'rr', 'gen'], help='Task name')
        parser.add_argument('--max_his', type=int, default=10, help='Max history length')
        
        parser.add_argument('--openrouter', type=str, help='Use OpenRouter with specified model (e.g., --openrouter google/gemini-2.0-flash-001)')
        parser.add_argument('--ollama', type=str, help='Use Ollama with specified model (e.g., --ollama llama3.2:1b)')
        parser.add_argument('--disable-reflection-rerun', action='store_false', dest='enable_reflection_rerun', help='Disable automatic rerun when reflector returns correctness: false (only for ReWOO system)')
        
        return parser

    def get_data(self, data_file: str, max_his: int) -> pd.DataFrame:
        df = pd.read_csv(data_file)
        
        # Handle missing 'history' column
        if 'history' not in df.columns:
            df['history'] = 'None'
        else:
            df['history'] = df['history'].fillna('None')
            df['history'] = df['history'].apply(lambda x: '\n'.join(x.split('\n')[-max_his:]))
        
        # Handle missing 'user_profile' column
        if 'user_profile' not in df.columns:
            df['user_profile'] = 'None'
        
        if self.task == 'sr':
            candidate_example: str = df['candidate_item_attributes'][0]
            self.n_candidate = len(candidate_example.split('\n'))
            self.system_kwargs['n_candidate'] = self.n_candidate  # Add n_candidate to system_kwargs by data sample
        return df

    def prompt_data(self, df: pd.DataFrame) -> list[tuple[str, int | float | str, pd.Series]]:
        data_prompt = self.system.prompts['data_prompt']
        if self.task == 'rp':
            return [(data_prompt.format(
                user_id=df['user_id'][i],
                user_profile=df['user_profile'][i],
                history=df['history'][i],
                target_item_id=df['item_id'][i],
                target_item_attributes=df['target_item_attributes'][i]
            ), df['rating'][i], df.iloc[i]) for i in tqdm(range(len(df)), desc="Loading data")]
        elif self.task == 'sr':
            candidate_example: str = df['candidate_item_attributes'][0]
            self.n_candidate = len(candidate_example.split('\n'))
            return [(data_prompt.format(
                user_id=df['user_id'][i],
                user_profile=df['user_profile'][i],
                history=df['history'][i],
                candidate_item_attributes=df['candidate_item_attributes'][i]
            ), df['item_id'][i], df.iloc[i]) for i in tqdm(range(len(df)), desc="Loading data") if df['rating'][i] >= 4]
        elif self.task == 'rr':
            # Retrieve & Rank: no candidate list in CSV; candidates must be provided
            # Set a default n_candidate for validation purposes
            self.system_kwargs['n_candidate'] = 6  # Default to 6 candidates for rr tasks
            return [(data_prompt.format(
                user_id=df['user_id'][i],
                user_profile=df['user_profile'][i],
                history=df['history'][i]
            ), df['item_id'][i], df.iloc[i]) for i in tqdm(range(len(df)), desc="Loading data") if df['rating'][i] >= 4]
        elif self.task == 'gen':
            return [(data_prompt.format(
                user_id=df['user_id'][i],
                user_profile=df['user_profile'][i],
                history=df['history'][i],
                target_item_id=df['item_id'][i],
                target_item_attributes=df['target_item_attributes'][i],
                rating=df['rating'][i]
            ), df['rating'][i], df.iloc[i]) for i in tqdm(range(len(df)), desc="Loading data")]
        else:
            raise NotImplementedError

    def get_system(self, system: str, system_config: str):
        if system == 'collaboration':
            self.system = CollaborationSystem(config_path=system_config, **self.system_kwargs)
        elif system == 'rewoo':
            self.system = ReWOOSystem(config_path=system_config, **self.system_kwargs)
        elif system in ['react', 'reflection', 'analyse']:
            raise NotImplementedError(f"System '{system}' has been deprecated and removed. Please use 'collaboration' or 'rewoo' instead.")
        else:
            raise NotImplementedError(f"Unknown system: {system}. Available systems: collaboration, rewoo")

    @property
    @abstractmethod
    def running_steps(self) -> int:
        """Return the steps to run for each trial.

        Raises:
            `NotImplementedError`: Subclasses should implement this method.
        Returns:
            `int`: The steps to run for each trial.
        """
        raise NotImplementedError

    @abstractmethod
    def before_generate(self) -> None:
        """The process to run before generating.

        Raises:
            `NotImplementedError`: Subclasses should implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def after_step(self, answer: Any, gt_answer: int | float | str, step: int, record: dict) -> None:
        """The process to run after each system step during one trial.

        Args:
            `answer` (`Any`): The answer given by the system.
            `gt_answer` (`int | float | str`): The ground truth answer.
            `step` (`int`): The current step. Starts from 0.
            `record` (`dict`): The record of the current trial. Can be used to store intermediate results.
        Raises:
            `NotImplementedError`: Subclasses should implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def after_iteration(self, answer: Any, gt_answer: int | float | str, record: dict, pbar: tqdm) -> None:
        """The process to run after each trial.

        Args:
            `answer` (`Any`): The final answer given by the system.
            `gt_answer` (`int | float | str`): The ground truth answer.
            `record` (`dict`): The record of the current trial. Can be used to store intermediate results.
            `pbar` (`tqdm`): The progress bar. Can be used to update the information of the progress bar.
        Raises:
            `NotImplementedError`: Subclasses should implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def after_generate(self) -> None:
        """The process to run after generating.

        Raises:
            `NotImplementedError`: Subclasses should implement this method.
        """
        raise NotImplementedError

    def generate(self, data: list[tuple[str, int | float | str, pd.Series]], steps: int = 2):
        # Start token and duration tracking for this generation task
        task_id = f"{self.dataset}_{self.task}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task_info = {
            'dataset': self.dataset,
            'task': self.task,
            'system': self.system.__class__.__name__,
            'model_override': self.model_override,
            'samples': len(data),
            'steps': steps,
            'max_history': self.max_his
        }
        
        token_tracker.start_task(task_id, task_info)
        duration_tracker.start_task(task_id, task_info)
        
        # Reset all agent LLM usage stats to start fresh
        token_tracker.reset_agent_stats(self.system)
        
        self.before_generate()
        with tqdm(total=len(data)) as pbar:
            for sample_idx, (test_data, gt_answer, data_sample) in enumerate(data):
                # Log sample progress (1-indexed for display and storage)
                sample_id = sample_idx + 1
                logger.info(f"Sample: {sample_id}/{len(data)}")
                
                record = dict()
                record['sample_id'] = sample_id
                record['user_id'] = data_sample.get('user_id', 'unknown')
                
                self.system.set_data(input=test_data, context="", gt_answer=gt_answer, data_sample=data_sample)
                # Store current sample_id for reflection improvement tracking (1-indexed)
                self.system._current_sample_idx = sample_id
                self.system._current_user_id = record['user_id']
                self.system.reset(clear=True)
                
                # NO LONGER reset agent LLM usage - let it accumulate for proper tracking
                # The token tracker will handle delta calculations
                
                for i in range(steps):
                    logger.debug(f'===================================Running step {i}...===================================')
                    self.after_step(answer=self.system(), gt_answer=gt_answer, step=i, record=record)
                    
                # Collect token stats after each sample
                token_tracker.collect_system_stats(self.system)
                
                self.after_iteration(answer=self.system.answer, gt_answer=gt_answer, record=record, pbar=pbar)
                pbar.update(1)
                
        # End token and duration tracking and save stats
        final_stats = token_tracker.end_task()
        duration_stats = duration_tracker.end_task()
        
        # Log summary
        logger.success("=== Token Usage Summary ===")
        logger.success(f"Task: {self.dataset} {self.task} ({len(data)} samples)")
        logger.success(f"Total API calls: {final_stats.get('total_api_calls', 0)}")
        logger.success(f"Total tokens: {final_stats.get('total_tokens', 0)}")
        logger.success(f"Input tokens: {final_stats.get('total_input_tokens', 0)}")
        logger.success(f"Output tokens: {final_stats.get('total_output_tokens', 0)}")
        logger.success(f"Models used: {final_stats.get('models_used', [])}")
        logger.success(f"Duration: {final_stats.get('duration', 0):.2f}s")
        
        # Log unified per-agent statistics
        agents = final_stats.get('agents', {})
        agent_durations = duration_stats.get('agents', {})
        
        if agents or agent_durations:
            logger.success("=== Per-Agent Statistics===")
            all_agent_names = set(agents.keys()) | set(agent_durations.keys())
            
            for agent_name in sorted(all_agent_names):
                logger.success(f"Agent: {agent_name}")
                
                # Token usage info
                if agent_name in agents:
                    agent_stats = agents[agent_name]
                    logger.success(f"  API calls: {agent_stats.get('api_calls', 0)}")
                    logger.success(f"  Total tokens: {agent_stats.get('total_tokens', 0)}")
                    logger.success(f"  Input tokens: {agent_stats.get('total_input_tokens', 0)}")
                    logger.success(f"  Output tokens: {agent_stats.get('total_output_tokens', 0)}")
                    logger.success(f"  Model: {agent_stats.get('model_name', 'unknown')}")
                
                # Execution duration info
                if agent_name in agent_durations:
                    duration_info = agent_durations[agent_name]
                    logger.success(f"  Total duration: {duration_info.get('total_duration', 0):.3f}s")
                    logger.success(f"  Number of calls: {duration_info.get('call_count', 0)}")
                    logger.success(f"  Average duration per call: {duration_info.get('avg_duration_per_call', 0):.3f}s")
        
        self.after_generate()


    def run(self, api_config: str, dataset: str, data_file: str, system: str, system_config: str, task: str, max_his: int, model: str = 'gemini', openrouter: str = None, ollama: str = None, enable_reflection_rerun: bool = True):
        if dataset == 'None':
            dataset = os.path.basename(os.path.dirname(data_file))
        self.dataset = dataset
        self.task = task
        self.max_his = max_his
        
        # Initialize API
        init_api(read_json(api_config))
        
        # Initialize system_kwargs early (before get_data which may modify it)
        # Only apply model override if explicitly specified via CLI
        if openrouter or ollama:
            # Determine model provider and setup
            provider_info = self._parse_provider_options(model, openrouter, ollama)
            self.model_override = provider_info['model_name']
            self.provider_type = provider_info['provider_type']
            
            self.system_kwargs = {
                'task': self.task,
                'leak': False,
                'dataset': self.dataset,
                'model_override': self.model_override,
                'provider_type': self.provider_type,
                'enable_reflection_rerun': enable_reflection_rerun,
            }
            
            logger.info(f"🤖 Using {provider_info['provider_type']} with model: {provider_info['model_name']}")
        else:
            # No CLI override - use individual agent configurations
            self.model_override = None
            self.provider_type = None
            
            self.system_kwargs = {
                'task': self.task,
                'leak': False,
                'dataset': self.dataset,
                'enable_reflection_rerun': enable_reflection_rerun,
            }
            
            logger.info("🤖 Using individual agent configurations from config files")
        
        # Load data (this may add n_candidate to system_kwargs for SR tasks)
        data_df = self.get_data(data_file, max_his)
        
        self.get_system(system, system_config)
        data = self.prompt_data(data_df)
        
        # Setup task-specific log file with actual sample count (after prompt_data which may filter samples)
        self.setup_task_logger(task=task, dataset=dataset, system=system, num_samples=len(data))
        
        self.generate(data, steps=self.running_steps)
    
    def _parse_provider_options(self, model: str, openrouter: str, ollama: str) -> dict:
        """Parse provider-specific options and return provider info."""
        # Count how many provider options are specified
        provider_count = sum(1 for x in [openrouter, ollama] if x is not None)
        
        if provider_count > 1:
            raise ValueError("Cannot specify multiple providers. Use either --openrouter OR --ollama, not both.")
        
        if openrouter:
            return {
                'provider_type': 'openrouter',
                'model_name': openrouter
            }
        elif ollama:
            return {
                'provider_type': 'ollama', 
                'model_name': ollama
            }
        else:
            # Use legacy --model parameter with OpenRouter as default
            return {
                'provider_type': 'openrouter',
                'model_name': model
            }
