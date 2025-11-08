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
from macrec.utils.prompt_builder import PromptBuilder
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
        parser.add_argument('--task', type=str, default='sr', choices=['rp', 'sr', 'gen'], help='Task name')
        parser.add_argument('--max_his', type=int, default=10, help='Max history length')
        
        parser.add_argument('--openrouter', type=str, help='Use OpenRouter with specified model (e.g., --openrouter google/gemini-2.0-flash-001)')
        parser.add_argument('--ollama', type=str, help='Use Ollama with specified model (e.g., --ollama llama3.2:1b)')
        parser.add_argument('--disable-reflection-rerun', action='store_false', dest='enable_reflection_rerun', help='Disable automatic rerun when reflector returns correctness: false (only for ReWOO system)')
        
        return parser

    def get_data(self, data_file: str, max_his: int) -> pd.DataFrame:
        """Load minimal CSV data and initialize prompt builder."""
        df = pd.read_csv(data_file)
        
        # Initialize prompt builder for on-demand text formatting
        data_dir = os.path.dirname(data_file)
        self.prompt_builder = PromptBuilder(data_dir, self.dataset)
        
        # For SR tasks, determine n_candidate from first row
        if self.task == 'sr' and 'candidate_item_id' in df.columns:
            # Parse first candidate list to get count
            import ast
            first_candidates = df['candidate_item_id'].iloc[0]
            if isinstance(first_candidates, str):
                try:
                    first_candidates = ast.literal_eval(first_candidates)
                except:
                    pass
            
            if isinstance(first_candidates, list):
                self.n_candidate = len(first_candidates)
                self.system_kwargs['n_candidate'] = self.n_candidate
                logger.info(f"Detected {self.n_candidate} candidates for SR task")
        
        return df

    def prompt_data(self, df: pd.DataFrame) -> list[tuple[str, int | float | str, pd.Series]]:
        """Build prompts on-demand from minimal CSV data."""
        import ast
        
        # First pass: Filter samples where GT not in candidates (BEFORE building prompts)
        if self.task == 'sr' and 'candidate_item_id' in df.columns:
            logger.info(f"Pre-filtering samples for {self.task} task...")
            valid_indices = []
            skipped_count = 0
            
            for i in range(len(df)):
                row = df.iloc[i]
                gt_item = row['item_id']
                candidate_ids = row['candidate_item_id']
                
                # Parse candidate list if it's a string
                if isinstance(candidate_ids, str):
                    try:
                        candidate_ids = ast.literal_eval(candidate_ids)
                    except:
                        logger.warning(f"Failed to parse candidate_item_id for sample {i+1}, skipping")
                        skipped_count += 1
                        continue
                
                # Check if GT in candidates
                if isinstance(candidate_ids, list):
                    if gt_item in candidate_ids:
                        valid_indices.append(i)
                    else:
                        logger.trace(f"Skipping sample {i+1} (User {row['user_id']}): GT item {gt_item} not in candidates")
                        skipped_count += 1
                else:
                    # If candidates not a list, include by default
                    valid_indices.append(i)
            
            # Filter dataframe to only valid samples
            if skipped_count > 0:
                logger.warning(f"Pre-filtered: Skipped {skipped_count}/{len(df)} samples where GT item not in candidates")
                df = df.iloc[valid_indices].reset_index(drop=True)
                logger.success(f"Building prompts for {len(df)} valid samples (filtered from {len(df) + skipped_count} total)")
            else:
                logger.info(f"All {len(df)} samples have GT in candidates")
        
        # Second pass: Build prompts only for valid samples
        data_prompt = self.system.prompts['data_prompt']
        prompts = []
        
        logger.info(f"Building prompts for {len(df)} samples...")
        
        for i in tqdm(range(len(df)), desc="Building prompts"):
            row = df.iloc[i]
            
            # Build formatted fields on-demand
            fields = self.prompt_builder.build_prompt_fields(row, max_his=self.max_his)
            
            # Build prompt based on task type
            if self.task == 'rp':
                prompt = data_prompt.format(
                    user_id=row['user_id'],
                    user_profile=fields['user_profile'],
                    history=fields['history'],
                    target_item_id=row['item_id'],
                    target_item_attributes=fields['target_item_attributes']
                )
                target = row['rating']
            
            elif self.task == 'sr':
                prompt = data_prompt.format(
                    user_id=row['user_id'],
                    user_profile=fields['user_profile'],
                    history=fields['history'],
                    candidate_item_attributes=fields['candidate_item_attributes']
                )
                target = row['item_id']
            
            elif self.task == 'gen':
                prompt = data_prompt.format(
                    user_id=row['user_id'],
                    user_profile=fields['user_profile'],
                    history=fields['history'],
                    target_item_id=row['item_id'],
                    target_item_attributes=fields['target_item_attributes'],
                    rating=row['rating']
                )
                target = row['rating']
            
            else:
                raise NotImplementedError(f"Task {self.task} not implemented")
            
            prompts.append((prompt, target, row))
        
        logger.info(f"Built {len(prompts)} prompts")
        return prompts

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
