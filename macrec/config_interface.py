"""
Configuration Interface for standardizing config handling across the system.
Provides consistent configuration management and validation.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import json
from pathlib import Path


class ConfigInterface(ABC):
    """Abstract interface for configuration management."""
    
    @abstractmethod
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure."""
        pass
    
    @abstractmethod
    def get_agent_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get agent configurations."""
        pass
    
    @abstractmethod
    def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration."""
        pass


class StandardConfigHandler(ConfigInterface):
    """Standard implementation of configuration interface."""
    
    def __init__(self, config_path: Optional[str] = None, config_data: Optional[Dict[str, Any]] = None):
        if config_path:
            self.config = self.load_config(config_path)
        elif config_data:
            self.config = config_data
        else:
            raise ValueError("Either config_path or config_data must be provided")
        
        self.validate_config(self.config)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration has required fields."""
        required_fields = ['supported_tasks', 'agents']
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Required field '{field}' missing from configuration")
        
        # Validate agents configuration
        if not isinstance(config['agents'], dict):
            raise ValueError("'agents' must be a dictionary")
        
        # Validate supported tasks
        if not isinstance(config['supported_tasks'], list):
            raise ValueError("'supported_tasks' must be a list")
        
        return True
    
    def get_agent_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get agent configurations."""
        return self.config.get('agents', {})
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get system-level configuration."""
        return {
            'supported_tasks': self.config.get('supported_tasks', []),
            'max_step': self.config.get('max_step', 10),
            'agent_prompt': self.config.get('agent_prompt'),
            'data_prompt': self.config.get('data_prompt'),
        }
    
    def get_agent_config(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for specific agent."""
        return self.config.get('agents', {}).get(agent_name)
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self.config.update(updates)
        self.validate_config(self.config)


class ToolConfigHandler:
    """Handler for tool configurations."""
    
    def __init__(self, tool_configs: Dict[str, str]):
        """Initialize with tool config file paths."""
        self.tool_configs = {}
        self.config_paths = tool_configs
        self._load_tool_configs()
    
    def _load_tool_configs(self):
        """Load all tool configurations."""
        for tool_name, config_path in self.config_paths.items():
            try:
                with open(config_path, 'r') as f:
                    self.tool_configs[tool_name] = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config for tool {tool_name}: {e}")
    
    def get_tool_config(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for specific tool."""
        return self.tool_configs.get(tool_name)
    
    def get_all_tool_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all tool configurations."""
        return self.tool_configs.copy()


class SystemConfigValidator:
    """Validates system configurations for consistency."""
    
    @staticmethod
    def validate_agent_dependencies(config: Dict[str, Any]) -> List[str]:
        """Validate agent dependencies are met."""
        warnings = []
        agents = config.get('agents', {})
        
        # Check for Manager requirement
        if 'Manager' not in agents:
            warnings.append("Manager agent is required but not configured")
        
        # Check ReWOO specific requirements
        if 'Planner' in agents and 'Solver' not in agents:
            warnings.append("Planner configured but Solver missing for ReWOO workflow")
        
        if 'Solver' in agents and 'Planner' not in agents:
            warnings.append("Solver configured but Planner missing for ReWOO workflow")
        
        return warnings
    
    @staticmethod
    def validate_task_support(config: Dict[str, Any], requested_task: str) -> bool:
        """Validate that requested task is supported."""
        supported_tasks = config.get('supported_tasks', [])
        return requested_task in supported_tasks
    
    @staticmethod
    def validate_file_paths(config: Dict[str, Any]) -> List[str]:
        """Validate that all referenced file paths exist."""
        missing_files = []
        
        def check_path(path_value, context):
            if isinstance(path_value, str) and path_value.endswith('.json'):
                if not Path(path_value).exists():
                    missing_files.append(f"{context}: {path_value}")
        
        # Check agent config paths
        for agent_name, agent_config in config.get('agents', {}).items():
            if 'config_path' in agent_config:
                check_path(agent_config['config_path'], f"Agent {agent_name} config_path")
            
            if 'prompt_config' in agent_config:
                check_path(agent_config['prompt_config'], f"Agent {agent_name} prompt_config")
            
            if 'thought_config_path' in agent_config:
                check_path(agent_config['thought_config_path'], f"Agent {agent_name} thought_config_path")
            
            if 'action_config_path' in agent_config:
                check_path(agent_config['action_config_path'], f"Agent {agent_name} action_config_path")
        
        # Check system-level paths
        if 'agent_prompt' in config:
            check_path(config['agent_prompt'], "System agent_prompt")
        
        if 'data_prompt' in config:
            check_path(config['data_prompt'], "System data_prompt")
        
        return missing_files