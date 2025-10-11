from argparse import ArgumentParser
import datetime

from macrec.tasks.base import Task
from macrec.systems import CollaborationSystem
from macrec.utils import init_api, read_json, token_tracker

class ChatTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--api_config', type=str, default='config/api-config.json', help='Api configuration file')
        parser.add_argument('--system', type=str, default='collaboration', choices=['collaboration'], help='System name')
        parser.add_argument('--system_config', type=str, required=True, help='System configuration file')
        parser.add_argument('--model', type=str, default=None, help='Override model name for all agents (e.g., gpt-4o for OpenRouter)')
        return parser

    def get_system(self, system: str, config_path: str, model_override: str = None):
        system_kwargs = {'task': 'chat'}
        if model_override:
            system_kwargs['model_override'] = model_override
            
        if system == 'collaboration':
            return CollaborationSystem(config_path=config_path, **system_kwargs)
        else:
            raise NotImplementedError(f"Unknown system: {system}. Use 'collaboration' for chat tasks.")

    def run(self, api_config: str, system: str, system_config: str, model: str = None, *args, **kwargs) -> None:
        init_api(read_json(api_config))
        
        # Setup task-specific log file for chat
        # For chat, we use 1 as the sample count since it's interactive
        self.setup_task_logger(task='chat', dataset='interactive', system=system, num_samples=1)
        
        self.system = self.get_system(system, system_config, model)
        
        # Start token tracking for chat session
        task_id = f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task_info = {
            'task_type': 'chat',
            'system': system,
            'system_config': system_config,
            'model_override': model
        }
        
        token_tracker.start_task(task_id, task_info)
        
        try:
            self.system.chat()
        finally:
            # End token tracking and collect stats
            token_tracker.collect_system_stats(self.system)
            final_stats = token_tracker.end_task()
            
            from loguru import logger
            logger.info("=== Chat Session Token Usage ===")
            logger.info(f"Total API calls: {final_stats.get('total_api_calls', 0)}")
            logger.info(f"Total tokens: {final_stats.get('total_tokens', 0)}")
            logger.info(f"Models used: {final_stats.get('models_used', [])}")
            logger.info(f"Duration: {final_stats.get('duration', 0):.2f}s")
            
            # Log per-agent statistics if available
            agents = final_stats.get('agents', {})
            if agents:
                logger.info("=== Per-Agent Token Usage ===")
                for agent_name, agent_stats in agents.items():
                    logger.info(f"Agent: {agent_name}")
                    logger.info(f"  API calls: {agent_stats.get('api_calls', 0)}")
                    logger.info(f"  Total tokens: {agent_stats.get('total_tokens', 0)}")
                    logger.info(f"  Input tokens: {agent_stats.get('total_input_tokens', 0)}")
                    logger.info(f"  Output tokens: {agent_stats.get('total_output_tokens', 0)}")
                    logger.info(f"  Model: {agent_stats.get('model_name', 'unknown')}")

if __name__ == '__main__':
    ChatTask().launch()
