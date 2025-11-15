from argparse import ArgumentParser
import datetime

from macrec.tasks.base import Task
from macrec.systems import ReWOOSystem
from macrec.utils import init_api, read_json, token_tracker

class ChatTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--api_config', type=str, default='config/api-config.json', help='Api configuration file')
        parser.add_argument('--system', type=str, default='rewoo', choices=['rewoo'], help='System name')
        parser.add_argument('--system_config', type=str, required=True, help='System configuration file')
        
        # Model configuration: use provider and model (model name)
        parser.add_argument('--provider', type=str, choices=['openrouter', 'openai', 'ollama', 'gemini'], help='LLM provider type (e.g., openrouter, openai, ollama, gemini)')
        parser.add_argument('--model', type=str, help='Model name/version to use (e.g., google/gemini-2.0-flash-001, gpt-4o-mini, llama3.2:1b). If not specified, uses default for the provider.')
        return parser

    def get_system(self, system: str, config_path: str, model_override: str = None, provider: str = None):
        system_kwargs = {'task': 'chat'}
        if model_override:
            system_kwargs['model_override'] = model_override
        if provider:
            system_kwargs['provider'] = provider

        if system == 'rewoo':
            return ReWOOSystem(config_path=config_path, **system_kwargs)
        else:
            raise NotImplementedError(f"Unknown system: {system}. Use 'rewoo' for chat tasks.")

    def run(self, api_config: str, system: str, system_config: str, provider: str = None, model: str = None, *args, **kwargs) -> None:
        init_api(read_json(api_config))
        
        provider_info = self._parse_provider_options(provider, model)
        model_override = provider_info.get('model')
        provider_value = provider_info.get('provider')
        
        # Setup task-specific log file for chat
        # For chat, we use 1 as the sample count since it's interactive
        self.setup_task_logger(task='chat', dataset='interactive', system=system, num_samples=1)
        
        self.system = self.get_system(system, system_config, model_override, provider_value)
        
        # Start token tracking for chat session
        task_id = f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task_info = {
            'task_type': 'chat',
            'system': system,
            'system_config': system_config,
            'model_override': model_override,
            'provider': provider_value
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
                    logger.info(f"  Model: {agent_stats.get('model', 'unknown')}")

    def _parse_provider_options(self, provider: str = None, model: str = None) -> dict:
        """Parse CLI options into provider metadata.
        
        Args:
            provider: Provider type (openrouter, openai, ollama, gemini)
            model: Optional model name. If not provided, uses default for the provider.
        
        Returns:
            dict with 'provider' and 'model' keys
        """
        def _get_default(provider_name: str) -> str:
            try:
                from macrec.utils import read_json
                api_config = read_json('config/api-config.json')
                if 'providers' in api_config and provider_name in api_config['providers']:
                    provider_cfg = api_config['providers'][provider_name]
                    for k in ('model', 'default_model'):
                        if provider_cfg.get(k):
                            return provider_cfg.get(k)
            except Exception:
                pass
            default_map = {
                'openrouter': 'google/gemini-2.0-flash-001',
                'openai': 'gpt-4o-mini',
                'ollama': 'llama3.2:1b',
                'gemini': 'google/gemini-2.0-flash-001'
            }
            return default_map.get(provider_name, 'google/gemini-2.0-flash-001')

        if not provider:
            return {'provider': None, 'model': None}
        
        # Use provided model or get default for the provider
        chosen_model = model if model else _get_default(provider)
        
        # Normalize model name for specific providers
        if provider == 'openai':
            chosen_model = self._normalize_openai_model(chosen_model)
        
        return {'provider': provider, 'model': chosen_model}

    @staticmethod
    def _normalize_openai_model(model: str) -> str:
        if not model:
            return 'gpt-4o-mini'
        cleaned = model.strip()
        if '/' in cleaned:
            prefix, suffix = cleaned.split('/', 1)
            if prefix.lower() == 'openai':
                return suffix
        return cleaned

if __name__ == '__main__':
    ChatTask().launch()
