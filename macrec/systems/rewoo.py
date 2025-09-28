import json
import re
from typing import Any, Dict, List, Optional
from loguru import logger

from macrec.systems.base import System
from macrec.agents import Agent, Manager, Analyst, Interpreter, Reflector, Searcher, Retriever, Planner, Solver
from macrec.utils import parse_answer, parse_action, format_chat_history


class ReWOOSystem(System):
    """
    ReWOO (Reasoning Without Observation) System for Multi-Agent Recommendation.
    
    This system implements the ReWOO pattern with three phases:
    1. Planning: Decompose complex tasks into sub-problems
    2. Working: Execute sub-tasks using existing agents
    3. Solving: Aggregate results to generate final recommendations
    
    The system maintains backward compatibility with existing CollaborationSystem
    while adding ReWOO capabilities for improved reasoning and performance.
    """
    
    def __init__(self, task: str, config_path: str, leak: bool = False, web_demo: bool = False, dataset: Optional[str] = None, *args, **kwargs) -> None:
        super().__init__(task, config_path, leak, web_demo, dataset, *args, **kwargs)
        
    def init(self, *args, **kwargs) -> None:
        """Initialize the ReWOO system."""
        self.max_step: int = self.config.get('max_step', 10)
        assert 'agents' in self.config, 'Agents are required.'
        self.init_agents(self.config['agents'])
        self.manager_kwargs = {
            'max_step': self.max_step,
            'task_type': self.task,
        }
        
        # Initialize ReWOO specific tracking
        self.analyzed_items = set()
        self.analyzed_users = set()
        self.execution_results = {}
        self.current_plan = None
        self.plan_steps = []
        self.step_n = 1
        self.phase = 'planning'

    @staticmethod
    def supported_tasks() -> list[str]:
        return ['rp', 'sr', 'rr', 'gen', 'chat']
    
    def init_agents(self, agents: dict[str, dict]) -> None:
        """Initialize agents with ReWOO support."""
        self.agents: dict[str, Agent] = dict()
        for agent, agent_config in agents.items():
            try:
                agent_class = globals()[agent]
                assert issubclass(agent_class, Agent), f'Agent {agent} is not a subclass of Agent.'
                
                # Apply model override if specified
                final_agent_config = agent_config.copy()
                if self.model_override:
                    # Handle different agent types
                    if agent == 'Manager':
                        # Manager has thought and action configs
                        thought_config = None
                        action_config = None
                        
                        if 'thought_config_path' in agent_config:
                            with open(agent_config['thought_config_path'], 'r') as f:
                                thought_config = json.load(f)
                        
                        if 'action_config_path' in agent_config:
                            with open(agent_config['action_config_path'], 'r') as f:
                                action_config = json.load(f)
                        
                        # Apply model override to both configs
                        if thought_config:
                            thought_config = self._apply_model_override(thought_config)
                            final_agent_config['thought_config'] = thought_config
                        
                        if action_config:
                            action_config = self._apply_model_override(action_config)
                            final_agent_config['action_config'] = action_config
                    
                    else:
                        # Other agents have a single config_path
                        if 'config_path' in agent_config:
                            with open(agent_config['config_path'], 'r') as f:
                                agent_llm_config = json.load(f)
                            
                            agent_llm_config = self._apply_model_override(agent_llm_config)
                            final_agent_config['config'] = agent_llm_config
                
                # Add prompt_config if specified in the agent config
                if 'prompt_config' in agent_config:
                    final_agent_config['prompt_config'] = agent_config['prompt_config']
                
                self.agents[agent] = agent_class(**final_agent_config, **self.agent_kwargs)
            except KeyError:
                raise ValueError(f'Agent {agent} is not supported.')
        
        # Ensure required agents for ReWOO
        if 'Planner' not in self.agents:
            logger.warning('Planner not configured. ReWOO will fall back to collaboration mode.')
        if 'Solver' not in self.agents:
            logger.warning('Solver not configured. ReWOO will fall back to collaboration mode.')

    @property
    def planner(self) -> Optional[Planner]:
        if 'Planner' not in self.agents:
            return None
        return self.agents['Planner']

    @property
    def solver(self) -> Optional[Solver]:
        if 'Solver' not in self.agents:
            return None
        return self.agents['Solver']

    @property
    def manager(self) -> Optional[Manager]:
        if 'Manager' not in self.agents:
            return None
        return self.agents['Manager']

    @property
    def analyst(self) -> Optional[Analyst]:
        if 'Analyst' not in self.agents:
            return None
        return self.agents['Analyst']

    @property
    def retriever(self) -> Optional[Retriever]:
        if 'Retriever' not in self.agents:
            return None
        return self.agents['Retriever']

    @property
    def searcher(self) -> Optional[Searcher]:
        if 'Searcher' not in self.agents:
            return None
        return self.agents['Searcher']

    @property
    def interpreter(self) -> Optional[Interpreter]:
        if 'Interpreter' not in self.agents:
            return None
        return self.agents['Interpreter']

    @property
    def reflector(self) -> Optional[Reflector]:
        if 'Reflector' not in self.agents:
            return None
        return self.agents['Reflector']

    def reset(self, clear: bool = False, preserve_progress: bool = False, *args, **kwargs) -> None:
        """Reset the ReWOO system state."""
        super().reset(clear, *args, **kwargs)
        
        if not preserve_progress:
            self.analyzed_items.clear()
            self.analyzed_users.clear()
            self.execution_results.clear()
            self.current_plan = None
            self.plan_steps = []
        
        self.step_n = 1
        self.phase = 'planning'

    def forward(self, user_input: Optional[str] = None, reset: bool = True) -> Any:
        """Forward method implementing ReWOO workflow with exact same data integration as collaboration system."""
        try:
            # Use exact same data integration as collaboration system
            if self.task == 'chat':
                self.manager_kwargs['history'] = self.chat_history if hasattr(self, 'chat_history') else []
            else:
                self.manager_kwargs['input'] = self.input
                
            if reset:
                self.reset()
                
            if self.task == 'chat':
                assert user_input is not None, 'User input is required for chat task.'
                if hasattr(self, 'add_chat_history'):
                    self.add_chat_history(user_input, role='user')
            
            # Check if ReWOO agents are available
            if self.planner is None or self.solver is None:
                # Fall back to collaboration mode if ReWOO agents not available
                logger.info("ReWOO agents not available. Falling back to collaboration mode.")
                return self._fallback_collaboration_mode()
            
            if self.phase == 'planning':
                return self._planning_phase()
            elif self.phase == 'working':
                return self._working_phase()
            elif self.phase == 'solving':
                return self._solving_phase()
            else:
                return self._planning_phase()  # Default to planning
                
        except Exception as e:
            logger.error(f"Error in ReWOO forward: {e}")
            # Fall back to collaboration mode on error
            return self._fallback_collaboration_mode()

    def _planning_phase(self) -> str:
        """Phase 1: Planning - Decompose task into sub-problems."""
        logger.info("ReWOO Phase 1: Planning")
        
        # Prepare query for planner
        query = self._prepare_planning_query()
        
        # Generate plan - CRITICAL: Pass manager_kwargs like collaboration system
        plan = self.planner.invoke(query, self.task, **self.manager_kwargs)
        self.current_plan = plan
        self.plan_steps = self.planner.parse_plan(plan)
        
        self.log(f"**ReWOO Plan Generated:**\n{plan}", agent=self.planner)
        
        # Move to working phase
        self.phase = 'working'
        self.step_n = 1
        
        if not self.plan_steps:
            # No valid steps found, proceed to solving with empty results
            self.phase = 'solving'
            return self._solving_phase()
        
        return self._working_phase()

    def _working_phase(self) -> str:
        """Phase 2: Working - Execute plan steps using workers."""
        logger.info(f"ReWOO Phase 2: Working - Step {self.step_n}")
        
        if self.step_n > len(self.plan_steps):
            # All steps completed, move to solving
            self.phase = 'solving'
            return self._solving_phase()
        
        # Get current step
        current_step = self.plan_steps[self.step_n - 1]
        
        # Check dependencies
        if not self._dependencies_satisfied(current_step):
            # Skip this step or handle dependency failure
            logger.warning(f"Dependencies not satisfied for step {self.step_n}")
            self.step_n += 1
            return self._working_phase()
        
        # Execute step
        result = self._execute_step(current_step)
        self.execution_results[current_step['variable']] = result
        
        self.log(f"**Step {self.step_n} ({current_step['variable']})**: {current_step['task_description']}\n**Result**: {result}", 
                agent=self._get_worker_agent(current_step['worker_type']))
        
        self.step_n += 1
        
        # Continue with next step or move to solving
        if self.step_n > len(self.plan_steps):
            self.phase = 'solving'
            return self._solving_phase()
        else:
            return self._working_phase()

    def _solving_phase(self) -> str:
        """Phase 3: Solving - Aggregate results to generate final answer."""
        logger.info("ReWOO Phase 3: Solving")
        
        # Generate final solution - CRITICAL: Pass manager_kwargs like collaboration system
        solution = self.solver.invoke(self.current_plan, self.execution_results, self.task, **self.manager_kwargs)
        
        self.log(f"**ReWOO Final Solution:**\n{solution}", agent=self.solver)
        
        # Extract and return final answer
        final_answer = self.solver.extract_final_answer(solution, self.task)
        
        # Add reflection capability if Reflector is available
        if self.reflector:
            logger.info("ReWOO Phase 3.1: Reflection")
            
            # Create a scratchpad-like summary of the entire ReWOO process for reflection
            rewoo_process = self._build_rewoo_scratchpad(solution, final_answer)
            
            # Use reflector to analyze the complete ReWOO process
            self.reflector(input=self.input, scratchpad=rewoo_process)
            
            if self.reflector.json_mode and self.reflector.reflections:
                try:
                    import json
                    reflection_json = json.loads(self.reflector.reflections[-1])
                    
                    # Log reflection results
                    if 'correctness' in reflection_json:
                        correctness = reflection_json['correctness']
                        reason = reflection_json.get('reason', 'No reason provided')
                        
                        if not correctness:
                            logger.warning(f"ReWOO Reflection identified issues: {reason}")
                            self.log(f"**ReWOO Reflection Issues Identified:**\n{reason}", agent=self.reflector)
                        else:
                            logger.info(f"ReWOO Reflection confirms correctness: {reason}")
                            self.log(f"**ReWOO Reflection Confirms Correctness:**\n{reason}", agent=self.reflector)
                            
                except Exception as e:
                    logger.error(f'Invalid reflection JSON output: {self.reflector.reflections[-1]}')
                    logger.error(f'JSON parsing error: {e}')
                    # Continue execution even if reflection parsing fails
            else:
                # Non-JSON mode reflection
                if self.reflector.reflections:
                    self.log(f"**ReWOO Reflection:**\n{self.reflector.reflections[-1]}", agent=self.reflector)
        
        # Log the final solution
        logger.info(f"ReWOO Final Answer: {final_answer}")
        
        return self.finish(final_answer)

    def _build_rewoo_scratchpad(self, solution: str, final_answer: str) -> str:
        """Build a comprehensive scratchpad of the ReWOO process for reflection."""
        scratchpad = f"\n=== ReWOO Process Summary ===\n"
        scratchpad += f"Task: {self.task.upper()}\n"
        scratchpad += f"Original Query: {getattr(self, 'input', 'No input')}\n\n"
        
        # Phase 1: Planning
        scratchpad += "=== Phase 1: Planning ===\n"
        if self.current_plan:
            scratchpad += f"Generated Plan:\n{self.current_plan}\n\n"
        else:
            scratchpad += "No plan generated - POTENTIAL ISSUE\n\n"
        
        # Phase 2: Working
        scratchpad += "=== Phase 2: Working (Execution Results) ===\n"
        if self.execution_results:
            for step_var, result in self.execution_results.items():
                scratchpad += f"{step_var}: {result}\n"
        else:
            scratchpad += "No execution results - POTENTIAL ISSUE\n"
        scratchpad += "\n"
        
        # Phase 3: Solving
        scratchpad += "=== Phase 3: Solving ===\n"
        scratchpad += f"Solver Output:\n{solution}\n"
        scratchpad += f"Final Answer: {final_answer}\n"
        
        return scratchpad

    def _prepare_planning_query(self) -> str:
        """Prepare the query for the planner based on task and context using exact same data as collaboration system."""
        # Use the same input data as collaboration system
        base_query = getattr(self, 'input', 'No input provided')
        
        if self.task == 'sr':
            return f"Sequential recommendation task: {base_query}"
        elif self.task == 'rp':
            return f"Rating prediction task: {base_query}" 
        elif self.task == 'rr':
            return f"Retrieve and rank task: {base_query}"
        elif self.task == 'gen':
            return f"Review generation task: {base_query}"
        else:
            return f"{self.task} task: {base_query}"

    def _dependencies_satisfied(self, step: Dict[str, Any]) -> bool:
        """Check if all dependencies for a step are satisfied."""
        dependencies = step.get('dependencies', [])
        variable = step.get('variable', 'unknown')
        
        for dep in dependencies:
            if dep not in self.execution_results:
                return False
        return True

    def _execute_step(self, step: Dict[str, Any]) -> str:
        """Execute a single plan step using the appropriate worker with real data exactly like collaboration system."""
        worker_type = step['worker_type']
        task_desc = step['task_description']
        
        # Replace dependency references with actual results
        for dep in step['dependencies']:
            if dep in self.execution_results:
                task_desc = task_desc.replace(dep, str(self.execution_results[dep]))
        
        # Get worker agent and execute using the same patterns as collaboration system
        worker = self._get_worker_agent(worker_type)
        if worker is None:
            return f"Worker {worker_type} not available"
        
        try:
            # Get json_mode setting - prefer manager's setting, but fallback to worker's own setting
            if self.manager and hasattr(self.manager, 'json_mode'):
                json_mode = self.manager.json_mode
            elif hasattr(worker, 'json_mode'):
                json_mode = worker.json_mode
            else:
                json_mode = False
            
            logger.debug(f"Worker {worker_type} json_mode: {json_mode}")
            
            if worker_type.lower() == 'analyst':
                # Use the task description to determine what to analyze
                args = self._parse_analyst_arguments_from_context(task_desc)
                logger.debug(f"Analyst args: {args}, type: {type(args)}")
                
                # Pass the full task description as analysis context
                kwargs = {
                    'task_context': task_desc,  # Pass the specific plan step description
                    **self.manager_kwargs
                }
                
                # Handle argument format based on json_mode
                if json_mode and isinstance(args, list):
                    # JSON mode expects list format
                    result = worker.invoke(argument=args, json_mode=json_mode, **kwargs)
                elif not json_mode and isinstance(args, list):
                    # Non-JSON mode expects string format
                    arg_string = f"{args[0]},{args[1]}" if len(args) >= 2 else "user,1"
                    result = worker.invoke(argument=arg_string, json_mode=json_mode, **kwargs)
                else:
                    # Pass as-is
                    result = worker.invoke(argument=args, json_mode=json_mode, **kwargs)
                    
            elif worker_type.lower() == 'retriever':
                # Use the same argument parsing as collaboration system  
                args = self._parse_retriever_arguments_from_context(task_desc)
                logger.debug(f"Retriever args: {args}, type: {type(args)}")
                # Only pass argument and json_mode like collaboration system does
                result = worker.invoke(argument=args, json_mode=json_mode)
            elif worker_type.lower() == 'searcher':
                logger.debug(f"Searcher args: {task_desc}, type: {type(task_desc)}")
                # Only pass argument and json_mode like collaboration system does
                result = worker.invoke(argument=task_desc, json_mode=json_mode)
            elif worker_type.lower() == 'interpreter':
                logger.debug(f"Interpreter args: {task_desc}, type: {type(task_desc)}")
                # Only pass argument and json_mode like collaboration system does
                result = worker.invoke(argument=task_desc, json_mode=json_mode)
            else:
                result = f"Unknown worker type: {worker_type}"
                
            # Track analyzed entities for compatibility (same as collaboration system)
            self._track_analyzed_entities(worker_type, task_desc, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing step with {worker_type}: {e}")
            return f"Execution failed: {str(e)}"

    def _get_worker_agent(self, worker_type: str) -> Optional[Agent]:
        """Get the appropriate worker agent."""
        worker_map = {
            'analyst': self.analyst,
            'retriever': self.retriever,
            'searcher': self.searcher,
            'interpreter': self.interpreter
        }
        return worker_map.get(worker_type.lower())

    def _parse_analyst_arguments_from_context(self, task_desc: str) -> List[Any]:
        """Parse analyst arguments from task description using real context data like collaboration system."""
        # Extract entity type and ID from the actual input context, same as collaboration system
        import re
        
        # First try to extract from task description
        user_match = re.search(r'user\s+(\d+)', task_desc, re.IGNORECASE)
        item_match = re.search(r'item\s+(\d+)', task_desc, re.IGNORECASE)
        
        if user_match:
            # Convert to integer as expected by Analyst
            return ['user', int(user_match.group(1))]
        elif item_match:
            # Convert to integer as expected by Analyst
            return ['item', int(item_match.group(1))]
        else:
            # Fall back to extracting from the actual input data like collaboration system
            input_text = getattr(self, 'input', '')
            user_match = re.search(r'user[_\s:]*(\d+)', input_text, re.IGNORECASE)
            item_match = re.search(r'item[_\s:]*(\d+)', input_text, re.IGNORECASE)
            
            if user_match:
                # Convert to integer as expected by Analyst
                return ['user', int(user_match.group(1))]
            elif item_match:
                # Convert to integer as expected by Analyst
                return ['item', int(item_match.group(1))]
            else:
                # Use kwargs like collaboration system
                if 'user_id' in self.kwargs:
                    return ['user', int(self.kwargs['user_id'])]
                elif 'item_id' in self.kwargs:
                    return ['item', int(self.kwargs['item_id'])]
                else:
                    # Default fallback
                    return ['user', 1]

    def _parse_retriever_arguments_from_context(self, task_desc: str) -> List[Any]:
        """Parse retriever arguments from task description using real context data like collaboration system."""
        import re
        
        # Extract user ID and number of candidates from context
        user_match = re.search(r'user\s+(\d+)', task_desc, re.IGNORECASE)
        num_match = re.search(r'(\d+)\s+(?:items?|candidates?)', task_desc, re.IGNORECASE)
        
        # Fall back to actual input data like collaboration system
        if not user_match:
            input_text = getattr(self, 'input', '')
            user_match = re.search(r'user[_\s:]*(\d+)', input_text, re.IGNORECASE)
        
        user_id = user_match.group(1) if user_match else str(self.kwargs.get('user_id', '1'))
        num_candidates = int(num_match.group(1)) if num_match else self.kwargs.get('n_candidate', 10)
        
        return [user_id, num_candidates]

    def _parse_analyst_arguments(self, task_desc: str) -> List[str]:
        """Parse analyst arguments from task description."""
        # Extract entity type and ID from description
        import re
        
        # Look for patterns like "analyze user 123" or "analyze item 456"
        user_match = re.search(r'user\s+(\d+)', task_desc, re.IGNORECASE)
        item_match = re.search(r'item\s+(\d+)', task_desc, re.IGNORECASE)
        
        if user_match:
            return ['user', user_match.group(1)]
        elif item_match:
            return ['item', item_match.group(1)]
        else:
            # Default case
            return ['item', str(self.kwargs.get('item_id', '1'))]

    def _parse_retriever_arguments(self, task_desc: str) -> List[Any]:
        """Parse retriever arguments from task description."""
        import re
        
        # Extract user ID and number of candidates
        user_match = re.search(r'user\s+(\d+)', task_desc, re.IGNORECASE)
        num_match = re.search(r'(\d+)\s+(?:items?|candidates?)', task_desc, re.IGNORECASE)
        
        user_id = user_match.group(1) if user_match else str(self.kwargs.get('user_id', '1'))
        num_candidates = int(num_match.group(1)) if num_match else self.kwargs.get('n_candidate', 10)
        
        return [user_id, num_candidates]

    def _track_analyzed_entities(self, worker_type: str, task_desc: str, result: str) -> None:
        """Track analyzed entities for compatibility with existing system."""
        if worker_type.lower() == 'analyst':
            import re
            user_match = re.search(r'user\s+(\d+)', task_desc, re.IGNORECASE)
            item_match = re.search(r'item\s+(\d+)', task_desc, re.IGNORECASE)
            
            if user_match:
                self.analyzed_users.add(user_match.group(1))
            elif item_match:
                self.analyzed_items.add(item_match.group(1))
        elif worker_type.lower() == 'retriever' and self.task == 'rr':
            # Extract item IDs from retriever response
            item_ids = re.findall(r'^(\d+):', result, re.MULTILINE)
            if item_ids:
                # Store retrieved items in kwargs for compatibility
                self.kwargs['retrieved_items'] = [int(id_str) for id_str in item_ids]

    def _fallback_collaboration_mode(self) -> str:
        """Fallback to collaboration system behavior using exact same execution pattern."""
        try:
            # If we have a manager, use collaboration system's step execution pattern
            if self.manager is not None:
                # Initialize scratchpad if needed (same as collaboration system)
                if not hasattr(self, 'scratchpad'):
                    self.scratchpad = ''
                
                # Use collaboration system's execution pattern
                if not self.is_finished() and not self.is_halted():
                    # Execute using collaboration system's think-act-execute pattern
                    self.think()
                    action_type, argument = self.act()
                    self.execute(action_type, argument)
                    self.step_n += 1
                    
                return "Collaboration mode step completed"
            else:
                # If no manager available, return error
                return "No manager available for fallback"
                
        except Exception as e:
            logger.error(f"Error in fallback collaboration mode: {e}")
            return f"Fallback failed: {str(e)}"
            
    def think(self):
        """Think method using exact same logic as collaboration system."""
        # Use the exact same implementation as collaboration system
        logger.debug(f'Step {self.step_n}:')
        logger.debug(f'Manager kwargs: {self.manager_kwargs}')
        
        # Truncate scratchpad if it's getting too long (same as collaboration)
        max_scratchpad_length = 8000
        if len(self.scratchpad) > max_scratchpad_length:
            lines = self.scratchpad.split('\n')
            truncated_lines = lines[-50:]
            self.scratchpad = '\n'.join(['[Previous context truncated...]'] + truncated_lines)
            logger.debug(f'Truncated scratchpad to prevent context overflow')
        
        self.scratchpad += f'\nThought {self.step_n}:'
        thought = self.manager(scratchpad=self.scratchpad, stage='thought', **self.manager_kwargs)
        
        # Clean up thought (same as collaboration system)
        thought = thought.split('Action')[0].split('Observation')[0].split('Thought')[0].strip()
        
        self.scratchpad += ' ' + thought
        self.log(f'**Thought {self.step_n}**: {thought}', agent=self.manager)

    def act(self) -> tuple[str, Any]:
        """Act method using exact same logic as collaboration system."""
        # Use exact same logic as collaboration system
        if self.max_step == self.step_n:
            self.scratchpad += f'\nHint: {self.manager.hint}'
        
        # Add progress reminder for rr tasks (same as collaboration)
        if self.task == 'rr' and hasattr(self, 'analyzed_items'):
            analyzed_count = len(self.analyzed_items)
            if 'retrieved_items' in self.manager_kwargs:
                remaining = [item for item in self.manager_kwargs['retrieved_items'] if item not in self.analyzed_items]
                if remaining:
                    progress_reminder = f'\nProgress: {analyzed_count}/10 items analyzed. Remaining: {sorted(remaining)}'
                else:
                    progress_reminder = f'\nProgress: {analyzed_count}/10 items analyzed.'
            else:
                progress_reminder = f'\nProgress: {analyzed_count}/10 items analyzed.'
            self.scratchpad += progress_reminder
        
        self.scratchpad += f'\nAction {self.step_n}:'
        logger.debug(f'Action step - Manager kwargs: {self.manager_kwargs}')
        action = self.manager(scratchpad=self.scratchpad, stage='action', **self.manager_kwargs)
        
        # Clean up action (same as collaboration system)
        action_clean = action.strip()
        if '```' in action_clean:
            parts = action_clean.split('```')
            for part in parts:
                if part.strip().startswith('{') and part.strip().endswith('}'):
                    action_clean = part.strip()
                    break
        
        # Take only the first JSON object if multiple are present
        if action_clean.count('{') > 1:
            brace_count = 0
            end_idx = 0
            for i, char in enumerate(action_clean):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            action_clean = action_clean[:end_idx]
        
        self.scratchpad += ' ' + action_clean
        from macrec.utils import parse_action
        action_type, argument = parse_action(action_clean, json_mode=self.manager.json_mode)
        logger.debug(f'Action {self.step_n}: {action_clean}')
        return action_type, argument

    def execute(self, action_type: str, argument: Any):
        """Execute method using exact same logic as collaboration system."""
        # Import the collaboration system to use its execute method
        from macrec.systems.collaboration import CollaborationSystem
        
        # Create a temporary instance to access the execute method
        # This ensures we use the exact same execution logic
        temp_collab = CollaborationSystem.__new__(CollaborationSystem)
        temp_collab.agents = self.agents
        temp_collab.manager_kwargs = self.manager_kwargs
        temp_collab.task = self.task
        temp_collab.kwargs = self.kwargs
        temp_collab.scratchpad = self.scratchpad
        temp_collab.step_n = self.step_n
        temp_collab.analyzed_items = self.analyzed_items
        temp_collab.analyzed_users = self.analyzed_users
        temp_collab.log = self.log
        temp_collab.finish = self.finish
        
        # Use collaboration system's execute method
        temp_collab.execute(action_type, argument)
        
        # Copy back the updated state
        self.scratchpad = temp_collab.scratchpad
        self.analyzed_items = temp_collab.analyzed_items
        self.analyzed_users = temp_collab.analyzed_users
        if hasattr(temp_collab, 'manager_kwargs'):
            self.manager_kwargs = temp_collab.manager_kwargs

    def is_finished(self) -> bool:
        """Check if the system is finished (same logic as collaboration system)."""
        return hasattr(self, 'finished') and self.finished

    def is_halted(self) -> bool:
        """Check if the system is halted (same logic as collaboration system)."""
        return ((self.step_n > self.max_step) or 
                (self.manager and self.manager.over_limit(scratchpad=getattr(self, 'scratchpad', ''), **self.manager_kwargs))) and not self.is_finished()

    def step(self):
        """Execute one step of the ReWOO process or fallback to collaboration."""
        try:
            # If in ReWOO mode and agents available, continue ReWOO workflow
            if self.planner and self.solver and self.phase in ['planning', 'working', 'solving']:
                return self.forward(reset=False)
            else:
                # Otherwise use collaboration system step pattern
                return self._fallback_collaboration_mode()
        except Exception as e:
            logger.error(f"Error in ReWOO step: {e}")
            # Fall back to collaboration mode on error
            return self._fallback_collaboration_mode()