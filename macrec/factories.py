"""
Agent Factory Pattern Implementation
Reduces coupling by providing centralized agent creation with dependency injection.
"""
from typing import Dict, Type, Any, Optional
from abc import ABC, abstractmethod

from macrec.agents.base import Agent
from macrec.agents import (
    Manager, Analyst, Reflector, 
    Planner, Solver
)


class AgentFactory(ABC):
    """Abstract factory for creating agents with dependency injection."""
    
    @abstractmethod
    def create_agent(self, agent_type: str, config: Dict[str, Any], **kwargs) -> Agent:
        """Create an agent of the specified type with given configuration."""
        pass


class DefaultAgentFactory(AgentFactory):
    """Default implementation of agent factory."""
    
    def __init__(self):
        self._agent_registry: Dict[str, Type[Agent]] = {
            'Manager': Manager,
            'Analyst': Analyst,
            'Reflector': Reflector,
            'Planner': Planner,
            'Solver': Solver,
        }
    
    def register_agent(self, name: str, agent_class: Type[Agent]) -> None:
        """Register a new agent type."""
        self._agent_registry[name] = agent_class
    
    def create_agent(self, agent_type: str, config: Dict[str, Any], **kwargs) -> Agent:
        """Create an agent of the specified type."""
        if agent_type not in self._agent_registry:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent_class = self._agent_registry[agent_type]
        return agent_class(**config, **kwargs)
    
    def get_available_agents(self) -> Dict[str, Type[Agent]]:
        """Get all registered agent types."""
        return self._agent_registry.copy()


class ToolProvider:
    """Provides tools to agents with dependency injection."""
    
    def __init__(self, tool_configs: Dict[str, Dict[str, Any]]):
        self.tool_configs = tool_configs
        self._tool_cache = {}
    
    def get_tool_config(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific tool."""
        return self.tool_configs.get(tool_name)
    
    def get_all_tool_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all tool configurations."""
        return self.tool_configs.copy()


class ConfigManager:
    """Centralized configuration management."""
    
    def __init__(self, config_data: Dict[str, Any]):
        self.config_data = config_data
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get system-level configuration."""
        return {
            'supported_tasks': self.config_data.get('supported_tasks', []),
            'max_step': self.config_data.get('max_step', 10),
            'agent_prompt': self.config_data.get('agent_prompt'),
            'data_prompt': self.config_data.get('data_prompt'),
        }
    
    def get_agent_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all agent configurations."""
        return self.config_data.get('agents', {})
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent."""
        return self.config_data.get('agents', {}).get(agent_name, {})