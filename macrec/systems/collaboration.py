import json
import re
from typing import Any, Optional, TYPE_CHECKING
from loguru import logger

from macrec.systems.base import System
from macrec.factories import DefaultAgentFactory, ConfigManager
from macrec.components import CollaborationOrchestrator, AgentCoordinator
from macrec.agents.base import Agent
from macrec.utils import parse_answer, parse_action, format_chat_history

if TYPE_CHECKING:
    from macrec.agents import Manager, Analyst, Interpreter, Reflector, Searcher, Retriever

class CollaborationSystem(System):
    def __init__(self, task: str, config_path: str, leak: bool = False, web_demo: bool = False, dataset: Optional[str] = None, *args, **kwargs) -> None:
        # Initialize state tracking for rr tasks first
        self.analyzed_items = set()  # Track which items have been analyzed
        self.analyzed_users = set()  # Track which users have been analyzed
        
        # Initialize factory and orchestrator components
        self.agent_factory = DefaultAgentFactory()
        self.agent_coordinator = AgentCoordinator(self.agent_factory)
        
        super().__init__(task, config_path, leak, web_demo, dataset, *args, **kwargs)
        self.action_history = []  # Track action history to detect loops
        self.max_repeated_actions = 3  # Maximum number of times the same action can be repeated
        self._chat_history = []  # Initialize chat history
        self.step_n = 1  # Initialize step counter

    @staticmethod
    def supported_tasks() -> list[str]:
        return ['rp', 'sr', 'rr', 'gen', 'chat']

    def init(self, *args, **kwargs) -> None:
        """
        Initialize the Collaboration system with factory pattern.
        """
        self.max_step: int = self.config.get('max_step', 10)
        assert 'agents' in self.config, 'Agents are required.'
        
        # Use agent coordinator to initialize agents
        self.agent_coordinator.initialize_agents(
            self.config['agents'], 
            **self.agent_kwargs
        )
        
        self.manager_kwargs = {
            'max_step': self.max_step,
            'task_type': self.task,
        }
        
        # Access agents through coordinator
        if self.agent_coordinator.get_agent('Reflector') is not None:
            self.manager_kwargs['reflections'] = ''
        if self.agent_coordinator.get_agent('Interpreter') is not None:
            self.manager_kwargs['task_prompt'] = ''
            
        # Initialize tracking sets for rr tasks
        self.analyzed_items = set()
        self.analyzed_users = set()

    @property
    def manager(self) -> Optional['Manager']:
        return self.agent_coordinator.get_agent('Manager')

    @property
    def analyst(self) -> Optional['Analyst']:
        return self.agent_coordinator.get_agent('Analyst')

    @property
    def interpreter(self) -> Optional['Interpreter']:
        return self.agent_coordinator.get_agent('Interpreter')

    @property
    def reflector(self) -> Optional['Reflector']:
        return self.agent_coordinator.get_agent('Reflector')

    @property
    def searcher(self) -> Optional['Searcher']:
        return self.agent_coordinator.get_agent('Searcher')

    @property
    def retriever(self) -> Optional['Retriever']:
        return self.agent_coordinator.get_agent('Retriever')

    def reset(self, clear: bool = False, preserve_progress: bool = False, *args, **kwargs) -> None:
        # Store progress state before reset if we're preserving progress
        preserved_scratchpad = ""
        preserved_analyzed_items = set()
        preserved_analyzed_users = set()
        preserved_kwargs = {}
        
        if preserve_progress:
            preserved_scratchpad = getattr(self, 'scratchpad', '')
            preserved_analyzed_items = getattr(self, 'analyzed_items', set()).copy()
            preserved_analyzed_users = getattr(self, 'analyzed_users', set()).copy()
            # Preserve important kwargs like n_candidate
            preserved_kwargs = {k: v for k, v in self.kwargs.items() if k in ['n_candidate']}
            logger.debug(f'Preserving progress: analyzed_items={preserved_analyzed_items}, analyzed_users={preserved_analyzed_users}')
        
        super().reset(*args, **kwargs)
        self.action_history = []  # Reset action history
        if clear:
            self._chat_history = []
        self._reset_action_history()
        
        # Reset state tracking for rr tasks, but preserve if requested
        if preserve_progress:
            # Restore preserved state
            self.scratchpad = preserved_scratchpad
            self.analyzed_items = preserved_analyzed_items
            self.analyzed_users = preserved_analyzed_users
            # Restore preserved kwargs
            self.kwargs.update(preserved_kwargs)
            logger.debug(f'Restored progress: analyzed_items={self.analyzed_items}, analyzed_users={self.analyzed_users}')
        else:
            # Normal reset - ensure sets exist and clear manager_kwargs from previous samples
            if not hasattr(self, 'analyzed_items'):
                self.analyzed_items = set()
            if not hasattr(self, 'analyzed_users'):
                self.analyzed_users = set()
            self.analyzed_items.clear()
            self.analyzed_users.clear()
            
            # Clear fields from manager_kwargs to prevent carryover between samples
            # Only clear fields that accumulate across samples, not ones set fresh each time
            fields_to_clear = ['retrieved_items', 'reflections', 'task_prompt']
            cleared_fields = []
            for field in fields_to_clear:
                if field in self.manager_kwargs:
                    del self.manager_kwargs[field]
                    cleared_fields.append(field)
            
            # Reset reflections to empty string if reflector exists
            if self.reflector is not None:
                self.manager_kwargs['reflections'] = ''
            
            if cleared_fields:
                logger.debug(f'Cleared {cleared_fields} from manager_kwargs for new sample')
            
        # Reset agents using agent coordinator
        self.agent_coordinator.reset_all_agents()

    def add_chat_history(self, chat: str, role: str) -> None:
        assert self.task == 'chat', 'Chat history is only available for chat task.'
        self._chat_history.append((chat, role))

    @property
    def chat_history(self) -> list[tuple[str, str]]:
        assert self.task == 'chat', 'Chat history is only available for chat task.'
        return format_chat_history(self._chat_history)

    def is_halted(self) -> bool:
        return ((self.step_n > self.max_step) or self.manager.over_limit(scratchpad=self.scratchpad, **self.manager_kwargs)) and not self.finished

    def _parse_answer(self, answer: Any = None) -> dict[str, Any]:
        if answer is None:
            answer = self.answer
        return parse_answer(type=self.task, answer=answer, gt_answer=self.gt_answer if self.task != 'chat' else '', json_mode=self.manager.json_mode, **self.kwargs)

    def think(self):
        # Think
        logger.debug(f'Step {self.step_n}:')
        logger.debug(f'Manager kwargs: {self.manager_kwargs}')
        
        # Truncate scratchpad if it's getting too long (prevent context overflow)
        max_scratchpad_length = 8000  # Adjust based on model context limit
        if len(self.scratchpad) > max_scratchpad_length:
            # Keep the last portion of the scratchpad with recent context
            lines = self.scratchpad.split('\n')
            # Keep approximately the last 50 lines
            truncated_lines = lines[-50:]
            self.scratchpad = '\n'.join(['[Previous context truncated...]'] + truncated_lines)
            logger.debug(f'Truncated scratchpad to prevent context overflow')
        
        self.scratchpad += f'\nThought {self.step_n}:'
        thought = self.manager(scratchpad=self.scratchpad, stage='thought', **self.manager_kwargs)
        
        # Clean up thought to prevent multiple actions/thoughts bleeding through
        thought = thought.split('Action')[0].split('Observation')[0].split('Thought')[0].strip()
        
        self.scratchpad += ' ' + thought
        self.log(f'**Thought {self.step_n}**: {thought}', agent=self.manager)

    def act(self) -> tuple[str, Any]:
        # Act
        if self.max_step == self.step_n:
            self.scratchpad += f'\nHint: {self.manager.hint}'
        
        # Add progress reminder for rr tasks
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
        
        # Removed confusing action examples that were causing hallucinations
        self.scratchpad += f'\nAction {self.step_n}:'
        logger.debug(f'Action step - Manager kwargs: {self.manager_kwargs}')
        action = self.manager(scratchpad=self.scratchpad, stage='action', **self.manager_kwargs)
        
        # Clean up action to get only the JSON part and prevent multiple actions
        action_clean = action.strip()
        if '```' in action_clean:
            # Extract JSON from code blocks
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
        action_type, argument = parse_action(action_clean, json_mode=self.manager.json_mode)
        logger.debug(f'Action {self.step_n}: {action_clean}')
        return action_type, argument

    def execute(self, action_type: str, argument: Any):
        # Execute
        log_head = ''
        
        # Track action for loop detection
        current_action = f"{action_type}:{str(argument)}"
        self.action_history.append(current_action)
        
        # Check for repeated actions (loop detection)
        if len(self.action_history) >= 3:
            recent_actions = self.action_history[-3:]
            if len(set(recent_actions)) == 1:  # Same action repeated 3 times
                # Only warn if it's not a finish action that might be retrying due to validation
                if action_type.lower() != 'finish':
                    observation = f'Warning: Repeated action "{action_type}" detected. Try a different approach.'
                    if self.task == 'rr':
                        # Convert all to strings to ensure consistent sorting
                        analyzed_items_str = sorted([str(x) for x in self.analyzed_items])
                        analyzed_users_str = sorted([str(x) for x in self.analyzed_users])
                        observation += f' For rr tasks, analyze each item only once. Analyzed: {analyzed_items_str}, Users: {analyzed_users_str}.'
                    log_head = ':red[Loop Detection]: '
                    self.scratchpad += f'\nObservation: {observation}'
                    logger.debug(f'Observation: {observation}')
                    self.log(f'{log_head}{observation}', agent=self.manager, logging=False)
                    return
        
        if action_type.lower() == 'finish':
            # For rr tasks, check if candidates have been retrieved
            if self.task == 'rr':
                if 'n_candidate' not in self.kwargs:
                    logger.debug(f'rr task: n_candidate not found in kwargs. Current kwargs: {self.kwargs}')
                    observation = 'For rr tasks, use Retrieve[user_id, 10] to get candidates before finishing.'
                    log_head = ':red[Error]: '
                else:
                    # Check if all retrieved items have been analyzed
                    expected_items = self.kwargs.get('n_candidate', 10)
                    if len(self.analyzed_items) < expected_items:
                        missing_items = expected_items - len(self.analyzed_items)
                        # Add detailed debugging information
                        debug_info = f' Analyzed items: {sorted(self.analyzed_items)}'
                        if 'retrieved_items' in self.manager_kwargs:
                            retrieved_items = self.manager_kwargs['retrieved_items']
                            remaining_items = [item for item in retrieved_items if item not in self.analyzed_items]
                            debug_info += f'. Remaining items to analyze: {sorted(remaining_items)}'
                        observation = f'For rr tasks, analyze ALL {expected_items} items before finishing. You have {len(self.analyzed_items)}/{expected_items}. Missing: {missing_items} items.{debug_info}'
                        log_head = ':red[Error]: '
                    else:
                        # All items analyzed, proceed with normal finish validation
                        parse_result = self._parse_answer(argument)
                        if parse_result['valid']:
                            observation = self.finish(parse_result['answer'])
                            log_head = ':violet[Finish with answer]:\n- '
                        else:
                            assert "message" in parse_result, "Invalid parse result."
                            observation = f'{parse_result["message"]} Valid Action examples are {self.manager.valid_action_example}.'
            else:
                # For non-rr tasks, proceed with normal finish validation
                parse_result = self._parse_answer(argument)
                if parse_result['valid']:
                    observation = self.finish(parse_result['answer'])
                    log_head = ':violet[Finish with answer]:\n- '
                else:
                    assert "message" in parse_result, "Invalid parse result."
                    observation = f'{parse_result["message"]} Valid Action examples are {self.manager.valid_action_example}.'
        elif action_type.lower() == 'analyse':
            if self.analyst is None:
                observation = 'Analyst is not configured. Cannot execute the action "Analyse".'
            else:
                # Track analyzed entities for rr tasks
                if self.task == 'rr':
                    if len(argument) >= 2:
                        entity_type, entity_id = argument[0], argument[1]
                        # Ensure consistent data types - always use string for tracking to avoid mixed types
                        entity_id = str(entity_id)
                        
                        if entity_type.lower() == 'item':
                            if entity_id in self.analyzed_items:
                                observation = f'Item {entity_id} has already been analyzed. Please analyze a different item or proceed to ranking.'
                                log_head = ':red[Warning]: '
                                self.scratchpad += f'\nObservation: {observation}'
                                logger.debug(f'Observation: {observation}')
                                self.log(f'{log_head}{observation}', agent=self.manager, logging=False)
                                return
                            else:
                                self.analyzed_items.add(entity_id)
                                logger.debug(f'Added item {entity_id} to analyzed_items. Total analyzed: {len(self.analyzed_items)}')
                        elif entity_type.lower() == 'user':
                            if entity_id in self.analyzed_users:
                                observation = f'User {entity_id} has already been analyzed. Please analyze a different user or proceed to ranking.'
                                log_head = ':red[Warning]: '
                                self.scratchpad += f'\nObservation: {observation}'
                                logger.debug(f'Observation: {observation}')
                                self.log(f'{log_head}{observation}', agent=self.manager, logging=False)
                                return
                            else:
                                self.analyzed_users.add(entity_id)
                                logger.debug(f'Added user {entity_id} to analyzed_users. Total analyzed: {len(self.analyzed_users)}')
                
                self.log(f':violet[Calling] :red[Analyst] :violet[with] :blue[{argument}]:violet[...]', agent=self.manager, logging=False)
                try:
                    observation = self.analyst.invoke(argument=argument, json_mode=self.manager.json_mode)
                    log_head = f':violet[Response from] :red[Analyst] :violet[with] :blue[{argument}]:violet[:]\n- '
                    
                    # Check if the analyst returned an error and suggest correction
                    if observation and "Invalid" in observation:
                        # Add a helpful hint to the scratchpad about the error
                        observation += " Please retry with the correct format in your next action."
                except Exception as e:
                    logger.error(f"Error in Analyst invocation: {e}")
                    observation = f"Analyst encountered an error: {str(e)}. Please try a different action or check your input format."
                    log_head = f':red[Error from] :red[Analyst] :red[with] :blue[{argument}]:red[:]\n- '
        elif action_type.lower() == 'search':
            if self.searcher is None:
                observation = 'Searcher is not configured. Cannot execute the action "Search".'
            else:
                self.log(f':violet[Calling] :red[Searcher] :violet[with] :blue[{argument}]:violet[...]', agent=self.manager, logging=False)
                try:
                    observation = self.searcher.invoke(argument=argument, json_mode=self.manager.json_mode)
                    log_head = f':violet[Response from] :red[Searcher] :violet[with] :blue[{argument}]:violet[:]\n- '
                except Exception as e:
                    logger.error(f"Error in Searcher invocation: {e}")
                    observation = f"Searcher encountered an error: {str(e)}. Please try a different search query."
                    log_head = f':red[Error from] :red[Searcher] :red[with] :blue[{argument}]:red[:]\n- '
        elif action_type.lower() == 'retrieve':
            if self.retriever is None:
                observation = 'Retriever is not configured. Cannot execute the action "Retrieve".'
            else:
                self.log(f':violet[Calling] :red[Retriever] :violet[with] :blue[{argument}]:violet[...]', agent=self.manager, logging=False)
                try:
                    observation = self.retriever.invoke(argument=argument, json_mode=self.manager.json_mode)
                    log_head = f':violet[Response from] :red[Retriever] :violet[with] :blue[{argument}]:violet[:]\n- '
                    
                    # For rr tasks, extract item IDs from retriever response to help with tracking
                    if self.task == 'rr' and observation:
                        # Parse the observation to extract item IDs and convert to integers for consistency
                        item_id_strings = re.findall(r'^(\d+):', observation, re.MULTILINE)
                        if item_id_strings:
                            # Convert to integers for consistent tracking
                            item_ids = [int(id_str) for id_str in item_id_strings]
                            logger.debug(f'Retrieved items for analysis tracking: {item_ids}')
                            # Add to manager kwargs for better context
                            if 'retrieved_items' not in self.manager_kwargs:
                                self.manager_kwargs['retrieved_items'] = item_ids
                except Exception as e:
                    logger.error(f"Error in Retriever invocation: {e}")
                    observation = f"Retriever encountered an error: {str(e)}. Please try a different retrieval query."
                    log_head = f':red[Error from] :red[Retriever] :red[with] :blue[{argument}]:red[:]\n- '
        elif action_type.lower() == 'interpret':
            if self.interpreter is None:
                observation = 'Interpreter is not configured. Cannot execute the action "Interpret".'
            else:
                self.log(f':violet[Calling] :red[Interpreter] :violet[with] :blue[{argument}]:violet[...]', agent=self.manager, logging=False)
                try:
                    observation = self.interpreter.invoke(argument=argument, json_mode=self.manager.json_mode)
                    log_head = f':violet[Response from] :red[Interpreter] :violet[with] :blue[{argument}]:violet[:]\n- '
                except Exception as e:
                    logger.error(f"Error in Interpreter invocation: {e}")
                    observation = f"Interpreter encountered an error: {str(e)}. Please try a different interpretation request."
                    log_head = f':red[Error from] :red[Interpreter] :red[with] :blue[{argument}]:red[:]\n- '
        else:
            observation = 'Invalid Action type or format. Valid Action examples are {self.manager.valid_action_example}.'

        self.scratchpad += f'\nObservation: {observation}'

        logger.debug(f'Observation: {observation}')
        self.log(f'{log_head}{observation}', agent=self.manager, logging=False)

    def step(self):
        try:
            self.think()
            action_type, argument = self.act()
            self.execute(action_type, argument)
            self.step_n += 1
        except Exception as e:
            logger.error(f"Error in step {self.step_n}: {e}")
            # Add error information to scratchpad to help system recover
            error_observation = f"System encountered an error: {str(e)}. Continuing with next action."
            self.scratchpad += f'\nObservation: {error_observation}'
            self.log(f':red[System Error]:red[:] {error_observation}', agent=self.manager, logging=False)
            self.step_n += 1
            
            # If we get too many consecutive errors, halt the system
            if hasattr(self, '_consecutive_errors'):
                self._consecutive_errors += 1
            else:
                self._consecutive_errors = 1
                
            if self._consecutive_errors >= 5:
                logger.error("Too many consecutive errors. Halting system.")
                self.halt("System halted due to excessive errors.")
            else:
                # Reset error counter on successful step
                if not str(e):  # If no error, reset counter
                    self._consecutive_errors = 0

    def reflect(self) -> bool:
        if (not self.is_finished() and not self.is_halted()) or self.reflector is None:
            self.reflected = False
            if self.reflector is not None:
                self.manager_kwargs['reflections'] = ''
            return False
        
        # Store current progress before reflection
        current_progress = {
            'analyzed_items': self.analyzed_items.copy(),
            'analyzed_users': self.analyzed_users.copy(),
            'n_candidate': self.kwargs.get('n_candidate'),
            'step_n': self.step_n
        }
        
        self.reflector(self.input, self.scratchpad)
        self.reflected = True
        self.manager_kwargs['reflections'] = self.reflector.reflections_str
        
        # Add progress context to reflections for the manager
        if current_progress['analyzed_items'] or current_progress['analyzed_users'] or current_progress['n_candidate']:
            progress_summary = f"\n\nPROGRESS SO FAR:\n"
            if current_progress['analyzed_users']:
                progress_summary += f"- Analyzed users: {sorted(current_progress['analyzed_users'])}\n"
            if current_progress['n_candidate']:
                progress_summary += f"- Retrieved {current_progress['n_candidate']} candidate items\n"
                # Show which items have been analyzed and which remain
                if 'retrieved_items' in self.manager_kwargs:
                    retrieved_items = self.manager_kwargs['retrieved_items']
                    analyzed_items = list(current_progress['analyzed_items'])  # Keep as integers
                    remaining_items = [item for item in retrieved_items if item not in analyzed_items]
                    if analyzed_items:
                        progress_summary += f"- Analyzed items: {sorted(analyzed_items)}\n"
                    if remaining_items:
                        progress_summary += f"- REMAINING items to analyze: {sorted(remaining_items)}\n"
                        progress_summary += f"- Next action should be: Analyse[item, {remaining_items[0]}]\n"
            elif current_progress['analyzed_items']:
                progress_summary += f"- Analyzed items: {sorted(current_progress['analyzed_items'])}\n"
            progress_summary += f"- Completed {current_progress['step_n']} steps\n"
            progress_summary += "IMPORTANT: Do not repeat the above analyses. Continue from where you left off.\n"
            self.manager_kwargs['reflections'] += progress_summary
        
        if self.reflector.json_mode:
            try:
                reflection_json = json.loads(self.reflector.reflections[-1])
                if 'correctness' in reflection_json and reflection_json['correctness']:
                    # don't forward if the last reflection is correct
                    logger.debug('Last reflection is correct, don\'t forward.')
                    self.log(":red[**Last reflection is correct, don't forward**]", agent=self.reflector, logging=False)
                    return False  # Return False to STOP reflecting (correct result)
            except Exception as e:
                logger.error(f'Invalid reflection JSON output: {self.reflector.reflections[-1]}')
                logger.error(f'JSON parsing error: {e}')
                # Continue execution even if reflection parsing fails
        return True  # Return True to CONTINUE reflecting (incorrect result)

    def interprete(self) -> None:
        if self.task == 'chat':
            assert self.interpreter is not None, 'Interpreter is required for chat task.'
            self.manager_kwargs['task_prompt'] = self.interpreter(input=self.chat_history)
        else:
            if self.interpreter is not None:
                self.manager_kwargs['task_prompt'] = self.interpreter(input=self.input)

    def forward(self, user_input: Optional[str] = None, reset: bool = True) -> Any:
        if self.task == 'chat':
            self.manager_kwargs['history'] = self.chat_history
        else:
            self.manager_kwargs['input'] = self.input
        if reset:
            self.reset()
        if self.task == 'chat':
            assert user_input is not None, 'User input is required for chat task.'
            self.add_chat_history(user_input, role='user')
        self.interprete()
        
        # Track reflection cycles to prevent infinite loops
        reflection_count = 0
        max_reflections = 3
        
        while not self.is_finished() and not self.is_halted():
            try:
                self.step()
                # Reset consecutive error counter on successful step
                if hasattr(self, '_consecutive_errors'):
                    self._consecutive_errors = 0
            except Exception as e:
                logger.error(f"Critical error in main loop: {e}")
                # Try to gracefully finish instead of crashing
                if hasattr(self, 'manager') and hasattr(self.manager, 'finish'):
                    try:
                        self.manager.finish(f"System encountered critical error: {str(e)}. Attempting graceful shutdown.")
                    except:
                        logger.error("Failed to gracefully finish. System will halt.")
                        break
                else:
                    break
        
        # Perform reflection after reasoning is complete
        should_continue_reflecting = self.reflect()
        while should_continue_reflecting and reflection_count < max_reflections:
            reflection_count += 1
            logger.debug(f'Starting reflection cycle {reflection_count}/{max_reflections}')
            
            # Reset with progress preservation for reflection continuation
            self.reset(preserve_progress=True)
            
            # Continue from where we left off
            while not self.is_finished() and not self.is_halted():
                self.step()
                
            # Check if we should continue reflecting
            should_continue_reflecting = self.reflect()
        
        if reflection_count >= max_reflections:
            logger.warning(f'Stopped after {max_reflections} reflection cycles to prevent infinite loops')
            
        if self.task == 'chat':
            self.add_chat_history(self.answer, role='system')
        return self.answer

    def chat(self) -> None:
        assert self.task == 'chat', 'Chat task is required for chat method.'
        print("Start chatting with the system. Type 'exit' or 'quit' to end the conversation.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            response = self(user_input=user_input, reset=True)
            print(f"System: {response}")

    def _reset_action_history(self):
        self.step_n: int = 1
        self.action_history = []
