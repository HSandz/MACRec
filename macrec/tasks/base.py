from abc import ABC, abstractmethod
from argparse import ArgumentParser
from loguru import logger
from typing import Any, Optional
import os
import datetime

from macrec.rl.reward import Reward, RatingPredictionRewardV1, RatingPredictionRewardV2, RatingPredictionReflectionReward, SequentialRecommendationRewardV1, SequentialRecommendationReflectionReward

class Task(ABC):
    def __init__(self):
        self.log_handler_id = None
        
    @staticmethod
    @abstractmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        """Parse task arguments.

        Args:
            `parser` (`ArgumentParser`): An `ArgumentParser` object to parse arguments.
        Raises:
            `NotImplementedError`: Should be implemented in subclasses.
        Returns:
            `ArgumentParser`: The `ArgumentParser` object with arguments added.
        """
        raise NotImplementedError

    def __getattr__(self, __name: str) -> Any:
        # return none if attribute not exists
        if __name not in self.__dict__:
            return None
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{__name}'")
    
    def setup_task_logger(self, task: str, dataset: str, system: str, num_samples: int):
        """Setup a task-specific log file with the naming convention:
        {task}_{dataset}_{system}_{samples}_{datetime}.log
        
        Args:
            task: Task type (e.g., 'sr', 'rp', 'gen')
            dataset: Dataset name (e.g., 'ml-100k', 'Beauty')
            system: System name (e.g., 'rewoo', 'collaboration')
            num_samples: Number of samples in the task
        """
        # Create task-specific log file
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"{task}_{dataset}_{system}_{num_samples}_{timestamp}.log"
        log_path = os.path.join("logs", log_filename)
        
        # Add a new file handler with the task-specific name
        self.log_handler_id = logger.add(log_path, level='INFO')
        logger.info(f"Task-specific log file: {log_path}")
        
        return log_path

    @abstractmethod
    def run(self, *args, **kwargs):
        """The running pipeline of the task.

        Raises:
            `NotImplementedError`: Should be implemented in subclasses.
        """
        raise NotImplementedError

    def _should_create_default_log(self) -> bool:
        """Check if this task type should create a default log file.
        
        Tasks that create their own specific log files (GenerationTask, ChatTask)
        should return False here.
        
        Returns:
            bool: True if a default log should be created, False otherwise
        """
        # Tasks that create their own log files
        task_with_custom_logs = ['GenerationTask', 'ChatTask', 'TestTask', 'EvaluateTask']
        return self.__class__.__name__ not in task_with_custom_logs
    
    def launch(self) -> Any:
        """Launch the task. Parse the arguments with `parse_task_args` and run the task with `run`. The parsed arguments are stored in `self.args` and passed to the `run` method.

        Returns:
            `Any`: The return value of the `run` method.
        """
        parser = ArgumentParser()
        parser = self.parse_task_args(parser)
        args, extras = parser.parse_known_args()
        self.args = args
        # log the arguments
        logger.success(args)
        
        # Setup a default log file for tasks that don't create their own
        # GenerationTask, ChatTask, TestTask, and EvaluateTask create task-specific logs
        if self._should_create_default_log() and self.log_handler_id is None:
            task_name = self.__class__.__name__.replace('Task', '').lower()
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f"{task_name}_{timestamp}.log"
            log_path = os.path.join("logs", log_filename)
            self.log_handler_id = logger.add(log_path, level='INFO')
            logger.info(f"Log file: {log_path}")
        
        return self.run(**vars(args))

class RewardTask(Task):
    def get_reward_model(self, reward_version: str) -> Reward:
        if self.task == 'rp':
            if reward_version == 'v1':
                return RatingPredictionRewardV1()
            elif reward_version == 'v2':
                return RatingPredictionRewardV2()
            elif reward_version == 'reflection':
                return RatingPredictionReflectionReward()
            else:
                raise NotImplementedError
        elif self.task == 'sr':
            if reward_version == 'v1':
                return SequentialRecommendationRewardV1()
            elif reward_version == 'reflection':
                return SequentialRecommendationReflectionReward()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
