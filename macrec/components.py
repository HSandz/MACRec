"""
System Components for breaking down large system classes.
Implements better separation of concerns and reduced coupling.
"""
from typing import Dict, Any, List, Optional, Set
from abc import ABC, abstractmethod
from loguru import logger

from macrec.agents.base import Agent
from macrec.factories import AgentFactory, ConfigManager


class SystemState:
    """Manages system state in a centralized way."""
    
    def __init__(self):
        self.step_count = 0
        self.max_steps = 10
        self.finished = False
        self.result = None
        self.chat_history = []
        self.action_history = []
        self.analyzed_items: Set[str] = set()
        self.analyzed_users: Set[str] = set()
        self.execution_context = {}
    
    def reset(self):
        """Reset system state."""
        self.step_count = 0
        self.finished = False
        self.result = None
        self.chat_history = []
        self.action_history = []
        self.analyzed_items.clear()
        self.analyzed_users.clear()
        self.execution_context.clear()
    
    def increment_step(self):
        """Increment step counter."""
        self.step_count += 1
    
    def is_max_steps_reached(self) -> bool:
        """Check if maximum steps reached."""
        return self.step_count >= self.max_steps
    
    def add_to_history(self, action: str, result: Any = None):
        """Add action to history."""
        self.action_history.append({
            'step': self.step_count,
            'action': action,
            'result': result
        })


class AgentCoordinator:
    """Coordinates communication and execution between agents."""
    
    def __init__(self, agent_factory: AgentFactory):
        self.agent_factory = agent_factory
        self.agents: Dict[str, Agent] = {}
    
    def initialize_agents(self, agent_configs: Dict[str, Dict[str, Any]], **kwargs):
        """Initialize agents from configuration with support for model overrides."""
        import json
        import os
        from loguru import logger
        
        self.agents.clear()
        
        # Extract system reference and model override info from kwargs
        system = kwargs.get('system')
        logger.debug(f"🔍 initialize_agents: system={system is not None}")
        model_override = getattr(system, 'model_override', None) if system else None
        
        for agent_name, config in agent_configs.items():
            try:
                # Apply model override if specified
                final_agent_config = config.copy()
                
                # Apply dataset substitution to agent config
                dataset = kwargs.get('dataset') or getattr(system, 'dataset', None)
                task = kwargs.get('task') or getattr(system, 'task', None)
                
                if dataset and 'config_path' in final_agent_config:
                    # Load the agent config file and apply substitutions
                    import json
                    with open(final_agent_config['config_path'], 'r') as f:
                        agent_config_content = f.read()
                    
                    # Substitute dataset and task placeholders
                    if dataset:
                        agent_config_content = agent_config_content.replace('{dataset}', dataset)
                    if task:
                        agent_config_content = agent_config_content.replace('{task}', task)
                    
                    # Parse the substituted config
                    substituted_config = json.loads(agent_config_content)
                    final_agent_config['config'] = substituted_config
                
                if model_override and system:
                    # Handle different agent types
                    if agent_name == 'Manager':
                        # Manager has thought and action configs
                        thought_config = None
                        action_config = None
                        
                        if 'thought_config_path' in config:
                            with open(config['thought_config_path'], 'r') as f:
                                thought_config = json.load(f)
                        
                        if 'action_config_path' in config:
                            with open(config['action_config_path'], 'r') as f:
                                action_config = json.load(f)
                        
                        # Apply model override to both configs
                        if thought_config:
                            thought_config = system._apply_model_override(thought_config)
                            final_agent_config['thought_config'] = thought_config
                        
                        if action_config:
                            action_config = system._apply_model_override(action_config)
                            final_agent_config['action_config'] = action_config
                    
                    else:
                        # Other agents have a single config_path
                        if 'config_path' in config:
                            with open(config['config_path'], 'r') as f:
                                agent_llm_config = json.load(f)
                            
                            agent_llm_config = system._apply_model_override(agent_llm_config)
                            final_agent_config['config'] = agent_llm_config
                
                agent = self.agent_factory.create_agent(agent_name, final_agent_config, **kwargs)
                self.agents[agent_name] = agent
                logger.info(f"Initialized agent: {agent_name}")
            except Exception as e:
                logger.error(f"Failed to initialize agent {agent_name}: {e}")
                raise
        
        # Ensure Manager is present for systems that require it
        system = kwargs.get('system')
        if system and 'Manager' not in self.agents:
            # ReWOO system doesn't require a Manager, but other systems do
            system_class_name = system.__class__.__name__
            if system_class_name != 'ReWOOSystem':
                raise ValueError(f'Manager is required for {system_class_name}.')
    
    def get_agent(self, agent_name: str) -> Optional[Agent]:
        """Get agent by name."""
        return self.agents.get(agent_name)
    
    def reset_all_agents(self):
        """Reset all agents."""
        for agent in self.agents.values():
            agent.reset()
    
    def execute_agent_action(self, agent_name: str, action: str, **kwargs) -> Any:
        """Execute an action on a specific agent."""
        agent = self.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent {agent_name} not found")
        
        try:
            if hasattr(agent, action):
                method = getattr(agent, action)
                return method(**kwargs)
            else:
                raise AttributeError(f"Agent {agent_name} has no action {action}")
        except Exception as e:
            logger.error(f"Error executing {action} on {agent_name}: {e}")
            raise


class SystemOrchestrator(ABC):
    """Abstract orchestrator for system workflow management."""
    
    def __init__(self, config_manager: ConfigManager, agent_coordinator: AgentCoordinator):
        self.config_manager = config_manager
        self.agent_coordinator = agent_coordinator
        self.state = SystemState()
    
    @abstractmethod
    def execute_workflow(self, **kwargs) -> Any:
        """Execute the system workflow."""
        pass
    
    def initialize(self, **kwargs):
        """Initialize the system."""
        system_config = self.config_manager.get_system_config()
        self.state.max_steps = system_config.get('max_step', 10)
        
        agent_configs = self.config_manager.get_agent_configs()
        self.agent_coordinator.initialize_agents(agent_configs, **kwargs)
    
    def reset(self):
        """Reset the system."""
        self.state.reset()
        self.agent_coordinator.reset_all_agents()
    
    def is_finished(self) -> bool:
        """Check if system execution is finished."""
        return self.state.finished or self.state.is_max_steps_reached()


class CollaborationOrchestrator(SystemOrchestrator):
    """Orchestrator for collaboration-based systems."""
    
    def execute_workflow(self, **kwargs) -> Any:
        """Execute collaboration workflow."""
        self.state.reset()
        
        while not self.is_finished():
            self.state.increment_step()
            
            try:
                # Get manager action
                manager = self.agent_coordinator.get_agent('Manager')
                if not manager:
                    raise ValueError("Manager agent not found")
                
                # Execute thinking step
                thought_result = manager.forward(mode='thought', **kwargs)
                
                # Execute action step
                action_result = manager.forward(mode='action', **kwargs)
                
                # Process action result and coordinate with other agents
                if self._should_finish(action_result):
                    self.state.finished = True
                    self.state.result = action_result
                    break
                
                # Execute action with appropriate agent
                action_type, arguments = self._parse_action(action_result)
                execution_result = self._execute_action(action_type, arguments, **kwargs)
                
                self.state.add_to_history(action_result, execution_result)
                
            except Exception as e:
                logger.error(f"Error in collaboration workflow step {self.state.step_count}: {e}")
                break
        
        return self.state.result
    
    def _should_finish(self, action_result: str) -> bool:
        """Check if the action indicates completion."""
        # Implementation would check for finish conditions
        return "Finish" in action_result or "FINISH" in action_result
    
    def _parse_action(self, action_result: str) -> tuple:
        """Parse action result to get action type and arguments."""
        # Implementation would parse the action string
        # This is a simplified version
        if "Search" in action_result:
            return ("search", action_result)
        elif "Analyze" in action_result:
            return ("analyze", action_result)
        else:
            return ("unknown", action_result)
    
    def _execute_action(self, action_type: str, arguments: str, **kwargs) -> Any:
        """Execute the parsed action with appropriate agent."""
        if action_type == "search":
            searcher = self.agent_coordinator.get_agent('Searcher')
            if searcher:
                return self.agent_coordinator.execute_agent_action('Searcher', 'forward', **kwargs)
        elif action_type == "analyze":
            analyst = self.agent_coordinator.get_agent('Analyst')
            if analyst:
                return self.agent_coordinator.execute_agent_action('Analyst', 'forward', **kwargs)
        
        return None


class ReWOOOrchestrator(SystemOrchestrator):
    """Orchestrator for ReWOO-style systems."""
    
    def execute_workflow(self, **kwargs) -> Any:
        """Execute ReWOO workflow (Plan -> Work -> Solve)."""
        self.state.reset()
        
        try:
            # Phase 1: Planning
            plan = self._execute_planning_phase(**kwargs)
            if not plan:
                return None
            
            # Phase 2: Working
            worker_results = self._execute_working_phase(plan, **kwargs)
            
            # Phase 3: Solving
            solution = self._execute_solving_phase(plan, worker_results, **kwargs)
            
            self.state.finished = True
            self.state.result = solution
            
        except Exception as e:
            logger.error(f"Error in ReWOO workflow: {e}")
            
        return self.state.result
    
    def _execute_planning_phase(self, **kwargs) -> Optional[str]:
        """Execute the planning phase."""
        planner = self.agent_coordinator.get_agent('Planner')
        if not planner:
            logger.error("Planner agent not found")
            return None
        
        return self.agent_coordinator.execute_agent_action('Planner', 'forward', **kwargs)
    
    def _execute_working_phase(self, plan: str, **kwargs) -> Dict[str, Any]:
        """Execute the working phase with multiple workers."""
        worker_results = {}
        
        # Parse plan to identify required workers
        # This is a simplified implementation
        workers = ['Analyst', 'Searcher']
        
        for worker_name in workers:
            worker = self.agent_coordinator.get_agent(worker_name)
            if worker:
                try:
                    result = self.agent_coordinator.execute_agent_action(
                        worker_name, 'forward', plan=plan, **kwargs
                    )
                    worker_results[worker_name] = result
                except Exception as e:
                    logger.warning(f"Worker {worker_name} failed: {e}")
        
        return worker_results
    
    def _execute_solving_phase(self, plan: str, worker_results: Dict[str, Any], **kwargs) -> Optional[str]:
        """Execute the solving phase."""
        solver = self.agent_coordinator.get_agent('Solver')
        if not solver:
            logger.error("Solver agent not found")
            return None
        
        return self.agent_coordinator.execute_agent_action(
            'Solver', 'forward', plan=plan, worker_results=worker_results, **kwargs
        )