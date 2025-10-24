"""Duration tracking utilities for monitoring agent execution time across tasks."""

from typing import Dict, List, Any, Optional
from loguru import logger
import time
from contextlib import contextmanager

class DurationTracker:
    """Centralized duration tracking for agent execution monitoring."""
    
    def __init__(self):
        self.task_stats: Dict[str, Dict] = {}
        self.current_task_id: Optional[str] = None
        self.agent_durations: Dict[str, float] = {}  # Track duration per agent per task
        self.agent_call_counts: Dict[str, int] = {}  # Track number of calls per agent
        
    def start_task(self, task_id: str, task_info: Dict[str, Any] = None) -> None:
        """Start tracking durations for a new task.
        
        Args:
            task_id (str): Unique identifier for the task
            task_info (Dict[str, Any], optional): Additional task information
        """
        self.current_task_id = task_id
        self.agent_durations = {}
        self.agent_call_counts = {}
        
        self.task_stats[task_id] = {
            'task_info': task_info or {},
            'agents': {},  # Will store per-agent duration stats
            'total_duration': 0,
        }
        
        logger.info(f"Started duration tracking for task: {task_id}")
        
    def end_task(self) -> Dict[str, Any]:
        """End the current task and return final stats.
        
        Returns:
            Dict[str, Any]: Final task statistics with agent durations
        """
        if self.current_task_id is None:
            logger.warning("No active task to end")
            return {}
            
        task_data = self.task_stats[self.current_task_id]
        
        # Aggregate all agent durations
        task_data['agents'] = {
            agent_name: {
                'total_duration': duration,
                'call_count': self.agent_call_counts.get(agent_name, 0),
                'avg_duration_per_call': duration / max(self.agent_call_counts.get(agent_name, 1), 1)
            }
            for agent_name, duration in self.agent_durations.items()
        }
        
        task_data['total_duration'] = sum(self.agent_durations.values())
        
        current_stats = task_data.copy()
        self.current_task_id = None
        return current_stats
        
    @contextmanager
    def track_agent_call(self, agent_name: str):
        """Context manager to track the duration of an agent call.
        
        Usage:
            with duration_tracker.track_agent_call("analyst"):
                # agent execution code
                result = agent.invoke(...)
        
        Args:
            agent_name (str): Name of the agent being tracked
        """
        if self.current_task_id is None:
            # If no task is being tracked, just yield without tracking
            yield
            return
            
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            # Accumulate duration for this agent
            if agent_name not in self.agent_durations:
                self.agent_durations[agent_name] = 0.0
                self.agent_call_counts[agent_name] = 0
                
            self.agent_durations[agent_name] += duration
            self.agent_call_counts[agent_name] += 1
            
            logger.debug(f"Agent '{agent_name}' call took {duration:.3f}s (total: {self.agent_durations[agent_name]:.3f}s, calls: {self.agent_call_counts[agent_name]})")
    
    def add_agent_duration(self, agent_name: str, duration: float) -> None:
        """Manually add a duration for an agent.
        
        Args:
            agent_name (str): Name of the agent
            duration (float): Duration in seconds
        """
        if self.current_task_id is None:
            return
            
        if agent_name not in self.agent_durations:
            self.agent_durations[agent_name] = 0.0
            self.agent_call_counts[agent_name] = 0
            
        self.agent_durations[agent_name] += duration
        self.agent_call_counts[agent_name] += 1
        
        logger.debug(f"Added {duration:.3f}s to agent '{agent_name}' (total: {self.agent_durations[agent_name]:.3f}s)")
    
    def get_agent_durations(self) -> Dict[str, float]:
        """Get current agent durations for the active task.
        
        Returns:
            Dict[str, float]: Dictionary mapping agent names to total duration in seconds
        """
        return self.agent_durations.copy()
    
    def get_agent_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed stats for all agents in the current task.
        
        Returns:
            Dict[str, Dict]: Dictionary with per-agent statistics
        """
        return {
            agent_name: {
                'total_duration': duration,
                'call_count': self.agent_call_counts.get(agent_name, 0),
                'avg_duration_per_call': duration / max(self.agent_call_counts.get(agent_name, 1), 1)
            }
            for agent_name, duration in self.agent_durations.items()
        }
    
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
        
    def reset(self) -> None:
        """Reset all tracking data."""
        self.task_stats = {}
        self.current_task_id = None
        self.agent_durations = {}
        self.agent_call_counts = {}
        logger.info("Duration tracker reset")

# Global duration tracker instance
duration_tracker = DurationTracker()
