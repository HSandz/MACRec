"""Token tracking utilities for monitoring LLM usage across tasks."""

from typing import Dict, List, Any
from loguru import logger
import json
import time

class TokenTracker:
    """Centralized token tracking for evaluation tasks."""
    
    def __init__(self):
        self.task_stats: Dict[str, Dict] = {}
        self.current_task_id: str = None
        self.start_time: float = None
        
    def start_task(self, task_id: str, task_info: Dict[str, Any] = None) -> None:
        """Start tracking a new task.
        
        Args:
            task_id (str): Unique identifier for the task
            task_info (Dict[str, Any], optional): Additional task information
        """
        self.current_task_id = task_id
        self.start_time = time.time()
        
        self.task_stats[task_id] = {
            'task_info': task_info or {},
            'start_time': self.start_time,
            'end_time': None,
            'duration': None,
            'agents': {},
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens': 0,
            'total_api_calls': 0,
            'models_used': set(),
        }
        
        logger.info(f"Started token tracking for task: {task_id}")
        
    def collect_agent_stats(self, agent_name: str, llm) -> None:
        """Collect token usage stats from an agent's LLM.
        
        Args:
            agent_name (str): Name of the agent
            llm: LLM instance with usage stats
        """
        if self.current_task_id is None:
            logger.warning("No active task for token tracking")
            return
            
        if not hasattr(llm, 'get_usage_stats'):
            logger.warning(f"LLM {llm.__class__.__name__} doesn't support usage tracking")
            return
            
        stats = llm.get_usage_stats()
        
        if stats['api_calls'] == 0:
            return  # No usage to track
            
        task_data = self.task_stats[self.current_task_id]
        
        # Store agent-specific stats
        task_data['agents'][agent_name] = stats.copy()
        
        # Update totals
        task_data['total_input_tokens'] += stats['total_input_tokens']
        task_data['total_output_tokens'] += stats['total_output_tokens']
        task_data['total_tokens'] += stats['total_tokens']
        task_data['total_api_calls'] += stats['api_calls']
        task_data['models_used'].add(stats['model_name'])
        
        logger.debug(f"Collected stats for {agent_name}: {stats['api_calls']} calls, {stats['total_tokens']} tokens")
        
    def collect_system_stats(self, system) -> None:
        """Collect token usage stats from all agents in a system.
        
        Args:
            system: System instance containing agents
        """
        if self.current_task_id is None:
            logger.warning("No active task for token tracking")
            return
            
        # Collect from different types of agents
        if hasattr(system, 'manager') and system.manager:
            if hasattr(system.manager, 'thought_llm'):
                self.collect_agent_stats('manager_thought', system.manager.thought_llm)
            if hasattr(system.manager, 'action_llm'):
                self.collect_agent_stats('manager_action', system.manager.action_llm)
                
        if hasattr(system, 'analyst') and system.analyst:
            if hasattr(system.analyst, 'analyst'):
                self.collect_agent_stats('analyst', system.analyst.analyst)
                
        if hasattr(system, 'reflector') and system.reflector:
            if hasattr(system.reflector, 'llm'):
                self.collect_agent_stats('reflector', system.reflector.llm)
                
        if hasattr(system, 'searcher') and system.searcher:
            if hasattr(system.searcher, 'searcher'):
                self.collect_agent_stats('searcher', system.searcher.searcher)
                
        if hasattr(system, 'interpreter') and system.interpreter:
            if hasattr(system.interpreter, 'interpreter'):
                self.collect_agent_stats('interpreter', system.interpreter.interpreter)
                
        if hasattr(system, 'retriever') and system.retriever:
            if hasattr(system.retriever, 'retriever_llm'):
                self.collect_agent_stats('retriever', system.retriever.retriever_llm)
                
        # Handle collaboration system with agents dict
        if hasattr(system, 'agents') and isinstance(system.agents, dict):
            for agent_name, agent in system.agents.items():
                if hasattr(agent, 'llm'):
                    self.collect_agent_stats(agent_name.lower(), agent.llm)
                elif hasattr(agent, 'thought_llm') and hasattr(agent, 'action_llm'):
                    self.collect_agent_stats(f'{agent_name.lower()}_thought', agent.thought_llm)
                    self.collect_agent_stats(f'{agent_name.lower()}_action', agent.action_llm)
                    
    def end_task(self) -> Dict[str, Any]:
        """End the current task and return final stats.
        
        Returns:
            Dict[str, Any]: Final task statistics
        """
        if self.current_task_id is None:
            logger.warning("No active task to end")
            return {}
            
        task_data = self.task_stats[self.current_task_id]
        task_data['end_time'] = time.time()
        task_data['duration'] = task_data['end_time'] - task_data['start_time']
        
        # Convert set to list for JSON serialization
        task_data['models_used'] = list(task_data['models_used'])
        
        logger.info(f"Task {self.current_task_id} completed:")
        logger.info(f"  Duration: {task_data['duration']:.2f}s")
        logger.info(f"  Total tokens: {task_data['total_tokens']}")
        logger.info(f"  API calls: {task_data['total_api_calls']}")
        logger.info(f"  Models used: {task_data['models_used']}")
        
        current_stats = task_data.copy()
        self.current_task_id = None
        return current_stats
        
    def get_task_stats(self, task_id: str = None) -> Dict[str, Any]:
        """Get statistics for a specific task.
        
        Args:
            task_id (str, optional): Task ID. If None, returns current task stats.
            
        Returns:
            Dict[str, Any]: Task statistics
        """
        if task_id is None:
            task_id = self.current_task_id
            
        if task_id is None or task_id not in self.task_stats:
            return {}
            
        return self.task_stats[task_id].copy()
        
    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics for all tasks.
        
        Returns:
            Dict[str, Dict]: All task statistics
        """
        return self.task_stats.copy()
        
    def save_stats(self, filepath: str) -> None:
        """Save all statistics to a JSON file.
        
        Args:
            filepath (str): Path to save the statistics
        """
        # Prepare data for JSON serialization
        data = {}
        for task_id, stats in self.task_stats.items():
            data[task_id] = stats.copy()
            # Ensure models_used is a list
            if isinstance(data[task_id]['models_used'], set):
                data[task_id]['models_used'] = list(data[task_id]['models_used'])
                
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        logger.info(f"Token tracking stats saved to: {filepath}")
        
    def reset(self) -> None:
        """Reset all tracking data."""
        self.task_stats = {}
        self.current_task_id = None
        self.start_time = None
        logger.info("Token tracker reset")

# Global token tracker instance
token_tracker = TokenTracker()
