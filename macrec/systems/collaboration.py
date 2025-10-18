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
    from macrec.agents import Manager, Analyst, Interpreter, Reflector, Searcher

class CollaborationSystem(System):
    def __init__(self, task: str, config_path: str, leak: bool = False, web_demo: bool = False, dataset: Optional[str] = None, *args, **kwargs) -> None:
        # Initialize tracking BEFORE super().__init__() because reset() needs it
        self.analyzed_items = set()  # Track analyzed items for SR task
        
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
        return ['rp', 'sr', 'gen', 'chat']

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

    def reset(self, clear: bool = False, preserve_progress: bool = False, *args, **kwargs) -> None:
        # Store progress state before reset if we're preserving progress
        preserved_scratchpad = ""
        preserved_kwargs = {}
        
        if preserve_progress:
            preserved_scratchpad = getattr(self, 'scratchpad', '')
            # Preserve important kwargs
            preserved_kwargs = {k: v for k, v in self.kwargs.items() if k in ['n_candidate', 'candidate_items']}
            logger.debug(f'Preserving progress')
        
        super().reset(*args, **kwargs)
        self.action_history = []  # Reset action history
        if clear:
            self._chat_history = []
        self._reset_action_history()
        
        # Reset state tracking, but preserve if requested
        if preserve_progress:
            # Restore preserved state
            self.scratchpad = preserved_scratchpad
            # Restore preserved kwargs
            self.kwargs.update(preserved_kwargs)
            logger.debug(f'Restored progress')
        else:
            # Reset analyzed items tracking
            self.analyzed_items.clear()
            
            # Clear fields from manager_kwargs to prevent carryover between samples
            fields_to_clear = ['reflections', 'task_prompt', 'candidate_items']
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
        
        # Add progress reminder for SR task
        if self.task == 'sr' and 'candidate_items' in self.manager_kwargs:
            total = len(self.manager_kwargs['candidate_items'])
            analyzed = len(self.analyzed_items)
            if analyzed < total:
                remaining = [item for item in self.manager_kwargs['candidate_items'] if item not in self.analyzed_items]
                # Show first 5 remaining items
                show_items = remaining[:5]
                more = f' (+{len(remaining)-5} more)' if len(remaining) > 5 else ''
                progress = f'\nProgress: {analyzed}/{total} items analyzed. Next: {show_items}{more}'
                self.scratchpad += progress
        
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
                    log_head = ':red[Loop Detection]: '
                    self.scratchpad += f'\nObservation: {observation}'
                    logger.debug(f'Observation: {observation}')
                    self.log(f'{log_head}{observation}', agent=self.manager, logging=False)
                    return
        
        if action_type.lower() == 'finish':
            # Validate SR task before finishing
            if self.task == 'sr' and 'candidate_items' in self.manager_kwargs:
                expected = len(self.manager_kwargs['candidate_items'])
                analyzed = len(self.analyzed_items)
                if analyzed < expected:
                    remaining = [item for item in self.manager_kwargs['candidate_items'] if item not in self.analyzed_items]
                    show_remaining = remaining[:5]
                    more = f' (+{len(remaining)-5} more)' if len(remaining) > 5 else ''
                    observation = f'You must analyze all {expected} candidate items before finishing. Analyzed: {analyzed}/{expected}. Remaining items: {show_remaining}{more}'
                    log_head = ':red[Validation Error]: '
                else:
                    parse_result = self._parse_answer(argument)
                    if parse_result['valid']:
                        observation = self.finish(parse_result['answer'])
                        log_head = ':violet[Finish with answer]:\n- '
                    else:
                        assert "message" in parse_result, "Invalid parse result."
                        observation = f'{parse_result["message"]} Valid Action examples are {self.manager.valid_action_example}.'
            else:
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
                # Track analyzed items for SR task
                if self.task == 'sr' and len(argument) >= 2:
                    entity_type, entity_id = argument[0], str(argument[1])
                    if entity_type.lower() == 'item':
                        if entity_id in self.analyzed_items:
                            observation = f'Item {entity_id} already analyzed. Analyze a different item.'
                            log_head = ':orange[Warning]: '
                            self.scratchpad += f'\nObservation: {observation}'
                            self.log(f'{log_head}{observation}', agent=self.manager, logging=False)
                            return
                        self.analyzed_items.add(entity_id)
                
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
            observation = 'Retriever agent has been removed from the system. Please use other available actions.'
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
        
        self.reflector(self.input, self.scratchpad)
        self.reflected = True
        self.manager_kwargs['reflections'] = self.reflector.reflections_str
        
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
        
        # Pass candidate_items to manager for SR task
        if self.task == 'sr' and 'candidate_items' in self.kwargs:
            self.manager_kwargs['candidate_items'] = self.kwargs['candidate_items']
            logger.debug(f'Passed {len(self.kwargs["candidate_items"])} candidate items to Manager')
        
        if reset:
            self.reset()
        if self.task == 'chat':
            assert user_input is not None, 'User input is required for chat task.'
            self.add_chat_history(user_input, role='user')
        self.interprete()
        
        # Track reflection cycles to prevent infinite loops
        reflection_count = 0
        max_reflections = 1
        
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
