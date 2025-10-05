from typing import Any, Dict
from loguru import logger

from macrec.agents.base import ToolAgent
from macrec.tools import InfoDatabase, InteractionRetriever
from macrec.utils import read_json, get_rm, parse_action

class Analyst(ToolAgent):
    def __init__(self, config_path: str = None, config: dict = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if config is not None:
            # Use provided config directly
            agent_config = config
        else:
            # Read config from file
            assert config_path is not None, "Either config_path or config must be provided"
            agent_config = read_json(config_path)
        
        tool_config: dict[str, dict] = get_rm(agent_config, 'tool_config', {})
        self.get_tools(tool_config)
        self.max_turns = get_rm(agent_config, 'max_turns', 15)  # Reduced default from 20 to 15
        self.analyst = self.get_LLM(config=agent_config)
        self.json_mode = self.analyst.json_mode
        # Track queried entities to prevent repetition
        self.queried_users = set()
        self.queried_items = set()
        self.gathered_info = {}
        # Store execution context for cached data access
        self.execution_context = None
        self.reset()

    @staticmethod
    def required_tools() -> dict[str, type]:
        return {
            'info_retriever': InfoDatabase,
            'interaction_retriever': InteractionRetriever,
        }

    @property
    def info_retriever(self) -> InfoDatabase:
        return self.tools['info_retriever']

    @property
    def interaction_retriever(self) -> InteractionRetriever:
        return self.tools['interaction_retriever']

    @property
    def analyst_prompt(self) -> str:
        if self.json_mode:
            return self.prompts['analyst_prompt_json']
        else:
            return self.prompts['analyst_prompt']

    @property
    def analyst_examples(self) -> str:
        if self.json_mode:
            return self.prompts['analyst_examples_json']
        else:
            return self.prompts['analyst_examples']

    @property
    def analyst_fewshot(self) -> str:
        if self.json_mode:
            return self.prompts['analyst_fewshot_json']
        else:
            return self.prompts['analyst_fewshot']

    @property
    def hint(self) -> str:
        if 'analyst_hint' not in self.prompts:
            return ''
        return self.prompts['analyst_hint']

    def _build_analyst_prompt(self, **kwargs) -> str:
        # Add command count and repetition warning to the prompt
        command_count = len(self._history)
        remaining_steps = self.max_turns - command_count
        
        # Check for recent repetitive patterns and provide context
        repetition_warning = ""
        if len(self._history) >= 2:
            recent_commands = [turn['command'] for turn in self._history[-2:]]
            if len(set(recent_commands)) == 1:  # All commands are the same
                repetition_warning = f"\nWARNING: You have been repeating the same command '{recent_commands[0]}'. This is wasteful and prohibited. You MUST try a different action or use Finish to complete the analysis."
        
        # Add explicit command history to show what was done
        command_summary = ""
        if len(self._history) > 0:
            unique_commands = []
            for turn in self._history:
                if turn['command'] not in unique_commands:
                    unique_commands.append(turn['command'])
            command_summary = f"\n\nCOMMANDS ALREADY EXECUTED:\n" + "\n".join([f"- {cmd}" for cmd in unique_commands])
            command_summary += "\n\nDO NOT REPEAT ANY OF THE ABOVE COMMANDS. Choose a different action or use Finish."
        
        # Add information about what has been queried already and what's cached
        context_info = ""
        if self.queried_users or self.queried_items:
            context_info += "\nAlready queried information:"
            if self.queried_users:
                context_info += f"\n- Users: {sorted(list(self.queried_users))}"
            if self.queried_items:
                context_info += f"\n- Items: {sorted(list(self.queried_items))}"
            context_info += "\nDo NOT query the same entities again unless absolutely necessary."
        
        # Add cached data information if execution context is available
        if self.execution_context and 'entity_cache' in self.execution_context:
            cache = self.execution_context['entity_cache']
            cached_items = []
            if cache.get('users'):
                cached_items.append(f"Users: {sorted(cache['users'].keys())}")
            if cache.get('items'):
                cached_items.append(f"Items: {sorted(cache['items'].keys())}")
            if cached_items:
                context_info += f"\n\nCACHED DATA AVAILABLE (from previous steps):\n- " + "\n- ".join(cached_items)
                context_info += "\nIMPORTANT: Use cached data instead of re-querying these entities."
        
        # Check if we have sufficient information to finish
        finish_hint = ""
        if len(self.gathered_info) >= 3:  # If we have info about 3+ entities
            finish_hint = f"\nYou have gathered information about {len(self.gathered_info)} entities. Consider if you have enough information to provide a meaningful analysis and use Finish command."
        
        # Add task context if provided (for ReWOO-specific analysis)
        task_context_info = ""
        
        # Debug logging to see what task_context we receive
        logger.debug(f"Analyst _build_analyst_prompt kwargs: {kwargs}")
        if 'task_context' in kwargs:
            logger.debug(f"Analyst task_context received: {kwargs['task_context']}")
        
        # First format the base prompt with shared content
        if 'task_context' in kwargs and kwargs['task_context']:
            # Replace the generic template with specific task context for ReWOO
            base_prompt_content = f"{kwargs['task_context']}\n\nCommands: UserInfo[id], ItemInfo[id], UserHistory[id], ItemHistory[id], Finish[result]\nGather 3-5 data points, avoid duplicates, then Finish.\n\nTarget: {kwargs.get('analyse_type', 'user')} {kwargs.get('id', '')}\n{self.history}"
            task_context_info = f"\n\nSPECIFIC FOCUS: {kwargs['task_context']}"
            logger.debug(f"Using task_context for base prompt: {kwargs['task_context']}")
        else:
            # Use the standard template
            base_prompt_content = self.prompts['analyst_base_prompt'].format(
                examples=self.analyst_examples,
                fewshot=self.analyst_fewshot,
                history=self.history,
                max_step=self.max_turns,
                hint=self.hint if len(self._history) + 1 >= self.max_turns else '',
                **kwargs
            )
            logger.debug("Using standard template, no task_context")
        
        # Then format the specific prompt (regular or JSON) using the base
        prompt = self.analyst_prompt.format(
            analyst_base_prompt=base_prompt_content
        )
        
        # Add step counter and repetition warning
        step_info = f"\nYou are at step {command_count + 1}/{self.max_turns}. Remaining steps: {remaining_steps}."
        if remaining_steps <= 3:
            step_info += " You should consider finishing your analysis soon."
        
        return prompt + context_info + finish_hint + task_context_info + step_info + repetition_warning + command_summary

    def _prompt_analyst(self, **kwargs) -> str:
        analyst_prompt = self._build_analyst_prompt(**kwargs)
        command = self.analyst(analyst_prompt)
        return command

    def command(self, command: str) -> None:
        logger.debug(f'Command: {command}')
        
        # Enhanced repetition detection
        if len(self._history) >= 2:
            # Check for exact repetition
            recent_commands = [turn['command'] for turn in self._history[-2:]]
            if all(cmd == command for cmd in recent_commands):
                # logger.info(f'Detected repetitive command: {command}. Forcing finish.')
                self.finish(f"Analysis completed. Detected repetitive pattern, ending analysis to prevent infinite loop.")
                return
            
            # Check for alternating pattern
            if len(self._history) >= 4:
                last_4_commands = [turn['command'] for turn in self._history[-4:]]
                if last_4_commands[0] == last_4_commands[2] and last_4_commands[1] == last_4_commands[3]:
                    # logger.info(f'Detected alternating repetitive commands: {last_4_commands}. Forcing finish.')
                    self.finish(f"Analysis completed. Detected alternating repetitive pattern, ending analysis.")
                    return
        
        # Only force finish if we have gathered too much information AND we're in a loop
        # This prevents premature termination while allowing natural LLM-driven completion
        if len(self.gathered_info) >= 7 and len(self._history) >= 8:
            # Check if we've been stuck without progress for several turns
            recent_actions = [parse_action(h['command'] if isinstance(h, dict) and 'command' in h else str(h), json_mode=self.json_mode)[0] 
                             for h in self._history[-4:] if h and (isinstance(h, dict) or str(h).strip())]
            if len(set(recent_actions)) <= 2:  # Only if stuck in repetitive pattern
                self.finish(f"Analysis completed. Gathered sufficient information about {len(self.gathered_info)} entities.")
                return
        
        log_head = ''
        action_type, argument = parse_action(command, json_mode=self.json_mode)
        
        if action_type.lower() == 'userinfo':
            try:
                if argument is None:
                    raise ValueError("Argument cannot be None")
                query_user_id = int(argument)
                
                # Check execution context cache first (from previous steps)
                if (self.execution_context and 'entity_cache' in self.execution_context and
                    query_user_id in self.execution_context['entity_cache'].get('users', {})):
                    cached_info = self.execution_context['entity_cache']['users'][query_user_id].get('info', '')
                    if cached_info:
                        observation = cached_info
                        self.queried_users.add(query_user_id)
                        self.gathered_info[f"user_{query_user_id}"] = observation
                        log_head = f':green[Used CACHED UserInfo for user] :red[{query_user_id}]:green[ (saved API call)]\n- '
                        logger.info(f"✓ Used cached data for user {query_user_id}, saved 1 API call")
                # Check if already queried in current session
                elif query_user_id in self.queried_users:
                    observation = f"User {query_user_id} information already retrieved. Use gathered information instead."
                    log_head = f':orange[Skipped duplicate UserInfo query for user] :red[{query_user_id}]:orange[...]\n- '
                else:
                    observation = self.info_retriever.user_info(user_id=query_user_id)
                    self.queried_users.add(query_user_id)
                    self.gathered_info[f"user_{query_user_id}"] = observation
                    log_head = f':violet[Look up UserInfo of user] :red[{query_user_id}]:violet[...]\n- '
            except (ValueError, TypeError):
                observation = f"Invalid user id: {argument}. Please provide a valid user ID number."
                log_head = ':red[Invalid UserInfo command]:red[...]\n- '
        elif action_type.lower() == 'iteminfo':
            try:
                if argument is None:
                    raise ValueError("Argument cannot be None")
                query_item_id = int(argument)
                
                # Check execution context cache first (from previous steps)
                if (self.execution_context and 'entity_cache' in self.execution_context and
                    query_item_id in self.execution_context['entity_cache'].get('items', {})):
                    cached_info = self.execution_context['entity_cache']['items'][query_item_id].get('info', '')
                    if cached_info:
                        observation = cached_info
                        self.queried_items.add(query_item_id)
                        self.gathered_info[f"item_{query_item_id}"] = observation
                        log_head = f':green[Used CACHED ItemInfo for item] :red[{query_item_id}]:green[ (saved API call)]\n- '
                        logger.info(f"✓ Used cached data for item {query_item_id}, saved 1 API call")
                # Check if already queried in current session
                elif query_item_id in self.queried_items:
                    observation = f"Item {query_item_id} information already retrieved. Use gathered information instead."
                    log_head = f':orange[Skipped duplicate ItemInfo query for item] :red[{query_item_id}]:orange[...]\n- '
                else:
                    observation = self.info_retriever.item_info(item_id=query_item_id)
                    self.queried_items.add(query_item_id)
                    self.gathered_info[f"item_{query_item_id}"] = observation
                    log_head = f':violet[Look up ItemInfo of item] :red[{query_item_id}]:violet[...]\n- '
            except (ValueError, TypeError):
                observation = f"Invalid item id: {argument}. Please provide a valid item ID number."
                log_head = ':red[Invalid ItemInfo command]:red[...]\n- '
        elif action_type.lower() == 'userhistory':
            try:
                if argument is None:
                    raise ValueError("Argument cannot be None")
                query_user_id = int(argument)
                history_key = f"user_history_{query_user_id}"
                
                # Check execution context cache first (from previous steps)
                if (self.execution_context and 'entity_cache' in self.execution_context and
                    query_user_id in self.execution_context['entity_cache'].get('user_histories', {})):
                    cached_history = self.execution_context['entity_cache']['user_histories'][query_user_id]
                    observation = cached_history
                    self.gathered_info[history_key] = observation
                    log_head = f':green[Used CACHED UserHistory for user] :red[{query_user_id}]:green[ (saved API call)]\n- '
                    logger.info(f"✓ Used cached history for user {query_user_id}, saved 1 API call")
                # Check if already queried in current session
                elif history_key in self.gathered_info:
                    observation = f"User {query_user_id} history already retrieved. Use gathered information instead."
                    log_head = f':orange[Skipped duplicate UserHistory query for user] :red[{query_user_id}]:orange[...]\n- '
                else:
                    # Use default k=10 for history retrieval
                    observation = self.interaction_retriever.user_retrieve(user_id=query_user_id, k=10)
                    self.gathered_info[history_key] = observation
                    log_head = f':violet[Look up UserHistory of user] :red[{query_user_id}]:violet[...]\n- '
            except (ValueError, TypeError):
                observation = f"Invalid user id: {argument}. Please provide a valid user ID number."
                log_head = ':red[Invalid UserHistory command]:red[...]\n- '
        elif action_type.lower() == 'itemhistory':
            try:
                if argument is None:
                    raise ValueError("Argument cannot be None")
                query_item_id = int(argument)
                history_key = f"item_history_{query_item_id}"
                
                # Check execution context cache first (from previous steps)
                if (self.execution_context and 'entity_cache' in self.execution_context and
                    query_item_id in self.execution_context['entity_cache'].get('item_histories', {})):
                    cached_history = self.execution_context['entity_cache']['item_histories'][query_item_id]
                    observation = cached_history
                    self.gathered_info[history_key] = observation
                    log_head = f':green[Used CACHED ItemHistory for item] :red[{query_item_id}]:green[ (saved API call)]\n- '
                    logger.info(f"✓ Used cached history for item {query_item_id}, saved 1 API call")
                # Check if already queried in current session
                elif history_key in self.gathered_info:
                    observation = f"Item {query_item_id} history already retrieved. Use gathered information instead."
                    log_head = f':orange[Skipped duplicate ItemHistory query for item] :red[{query_item_id}]:orange[...]\n- '
                else:
                    # Use default k=10 for history retrieval
                    observation = self.interaction_retriever.item_retrieve(item_id=query_item_id, k=10)
                    self.gathered_info[history_key] = observation
                    log_head = f':violet[Look up ItemHistory of item] :red[{query_item_id}]:violet[...]\n- '
            except (ValueError, TypeError):
                observation = f"Invalid item id: {argument}. Please provide a valid item ID number."
                log_head = ':red[Invalid ItemHistory command]:red[...]\n- '
        elif action_type.lower() == 'finish':
            # Handle various types of finish content
            if isinstance(argument, dict):
                # If argument is a dict, try to extract meaningful content
                if 'content' in argument:
                    finish_content = str(argument['content'])
                else:
                    # Convert dict to readable string
                    finish_content = str(argument)
            elif isinstance(argument, list):
                # If argument is a list, join elements
                finish_content = ', '.join(str(item) for item in argument)
            else:
                # Use argument as-is for strings and other types
                finish_content = str(argument) if argument is not None else "Analysis completed"
            
            # Generate detailed analysis based on gathered information
            detailed_analysis = self._generate_detailed_analysis(finish_content)
            observation = self.finish(results=detailed_analysis)
            log_head = ':violet[Finish with results]:\n- '
        else:
            observation = f'Unknown command type: {action_type}.'
        logger.debug(f'Observation: {observation}')
        self.observation(observation, log_head)
        turn = {
            'command': command,
            'observation': observation,
        }
        self._history.append(turn)

    def _generate_detailed_analysis(self, original_finish_content: str) -> str:
        """Generate detailed analysis based on gathered information."""
        if not self.gathered_info:
            # No information gathered, return original content
            return original_finish_content
        
        # Build comprehensive analysis from gathered data
        analysis_parts = []
        
        # Add original finish content if meaningful
        if original_finish_content and original_finish_content.strip().lower() != 'analysis':
            analysis_parts.append(f"Initial Analysis: {original_finish_content}")
        
        # Analyze gathered user information
        user_info_parts = []
        user_history_parts = []
        item_info_parts = []
        
        for key, value in self.gathered_info.items():
            if key.startswith('user_') and not key.startswith('user_history_'):
                user_id = key.replace('user_', '')
                user_info_parts.append(f"User {user_id}: {value}")
            elif key.startswith('user_history_'):
                user_id = key.replace('user_history_', '')
                user_history_parts.append(f"User {user_id} History: {value}")
            elif key.startswith('item_'):
                item_id = key.replace('item_', '')
                item_info_parts.append(f"Item {item_id}: {value}")
        
        # Compile analysis sections
        if user_info_parts:
            analysis_parts.append("User Information Analysis:")
            analysis_parts.extend([f"  - {part}" for part in user_info_parts])
        
        if user_history_parts:
            analysis_parts.append("User History Analysis:")
            analysis_parts.extend([f"  - {part}" for part in user_history_parts])
        
        if item_info_parts:
            analysis_parts.append("Item Information Analysis:")
            analysis_parts.extend([f"  - {part}" for part in item_info_parts])
        
        # Add summary insights
        if len(self.gathered_info) > 1:
            analysis_parts.append(f"Summary: Analyzed {len(self.gathered_info)} data points including user profiles, interaction histories, and item details to provide comprehensive recommendation insights.")
        
        # Join all parts
        detailed_analysis = "\n".join(analysis_parts)
        
        # Fallback to original if no meaningful analysis generated
        if not detailed_analysis.strip():
            return original_finish_content
        
        return detailed_analysis

    def forward(self, id: int, analyse_type: str, *args: Any, **kwargs: Any) -> str:
        assert self.system.data_sample is not None, "Data sample is not provided."
        assert 'user_id' in self.system.data_sample, "User id is not provided."
        assert 'item_id' in self.system.data_sample, "Item id is not provided."
        self.interaction_retriever.reset(user_id=self.system.data_sample['user_id'], item_id=self.system.data_sample['item_id'])
        
        consecutive_invalid_commands = 0
        max_invalid_commands = 3
        
        while not self.is_finished():
            try:
                command = self._prompt_analyst(id=id, analyse_type=analyse_type, **kwargs)
                
                # Ensure command is a string
                if not isinstance(command, str):
                    logger.error(f"LLM returned non-string response: {type(command)} - {command}")
                    if isinstance(command, dict):
                        # Try to extract content from dict response
                        command = command.get('content', str(command))
                    else:
                        command = str(command)
                
                # Handle JSON arrays of commands
                if self.json_mode and command.strip().startswith('['):
                    # Parse JSON array of commands
                    try:
                        import json
                        commands_array = json.loads(command.strip())
                        if isinstance(commands_array, list):
                            # Track commands in this array to prevent duplicates
                            executed_in_array = set()
                            # Execute each command in sequence
                            for cmd_obj in commands_array:
                                if isinstance(cmd_obj, dict) and 'type' in cmd_obj:
                                    # Convert back to JSON string for individual processing
                                    individual_command = json.dumps(cmd_obj)
                                    
                                    # Skip if already executed in this array
                                    if individual_command in executed_in_array:
                                        logger.info(f"Skipping duplicate command in array: {individual_command}")
                                        continue
                                    executed_in_array.add(individual_command)
                                    
                                    action_type, argument = parse_action(individual_command, json_mode=self.json_mode)
                                    
                                    if action_type.lower() == 'invalid':
                                        consecutive_invalid_commands += 1
                                        logger.warning(f"Invalid command in array: {individual_command}")
                                        continue
                                    else:
                                        consecutive_invalid_commands = 0
                                    
                                    self.command(individual_command)
                                    
                                    # Break if finished (Finish command was executed)
                                    if self.is_finished():
                                        break
                            continue  # Skip normal single command processing
                        else:
                            logger.warning(f"Expected JSON array but got: {type(commands_array)}")
                    except Exception as e:
                        logger.error(f"Error parsing JSON array: {e}")
                        # Fall back to single command processing
                
                # Normal single command processing
                action_type, argument = parse_action(command, json_mode=self.json_mode)
                
                if action_type.lower() == 'invalid':
                    consecutive_invalid_commands += 1
                    logger.warning(f"Invalid command generated: {command} (attempt {consecutive_invalid_commands})")
                    
                    if consecutive_invalid_commands >= max_invalid_commands:
                        logger.error(f"Too many consecutive invalid commands. Forcing finish.")
                        self.finish("Analysis terminated due to repeated invalid responses.")
                        break
                    
                    # Add error feedback to history to help the model learn
                    error_observation = f"Invalid command format: '{command}'. Please use the correct JSON format with valid values."
                    turn = {
                        'command': command,
                        'observation': error_observation,
                    }
                    self._history.append(turn)
                    continue
                else:
                    consecutive_invalid_commands = 0  # Reset counter on valid command
                
                self.command(command)
            except Exception as e:
                logger.error(f"Error in analyst forward: {e}")
                self.finish(f"Analysis terminated due to error: {str(e)}")
                break
                
        if not self.finished:
            return "Analyst did not return any result."
        return self.results

    def invoke(self, argument: Any, json_mode: bool, task_context: str = None, execution_context: Dict[str, Any] = None, **kwargs) -> str:
        """
        Invoke the analyst with specific arguments and optional task context.
        
        Args:
            argument: The analysis argument (analyse_type, id)
            json_mode: Whether to use JSON mode
            task_context: Optional specific task description for context-aware analysis
            execution_context: Optional context with cached data from previous steps
            **kwargs: Additional keyword arguments
        """
        # Store execution context for cache access
        self.execution_context = execution_context
        
        if json_mode:
            if not isinstance(argument, list) or len(argument) != 2:
                observation = "The argument of the action 'Analyse' should be a list with two elements: analyse type (user or item) and id."
                return observation
            else:
                analyse_type, id = argument
                if (isinstance(id, str) and 'user_' in id) or (isinstance(id, str) and 'item_' in id):
                    observation = f"Invalid id: {id}. Don't use the prefix 'user_' or 'item_'. Just use the id number only, e.g., 1, 2, 3, ..."
                    return observation
                elif analyse_type.lower() not in ['user', 'item']:
                    observation = f"Invalid analyse type: {analyse_type}. It should be either 'user' or 'item'."
                    return observation
                elif not isinstance(id, int):
                    observation = f"Invalid id: {id}. It should be an integer."
                    return observation
        else:
            if len(argument.split(',')) != 2:
                observation = "The argument of the action 'Analyse' should be a string with two elements separated by a comma: analyse type (user or item) and id."
                return observation
            else:
                analyse_type, id = argument.split(',')
                if 'user_' in id or 'item_' in id:
                    observation = f"Invalid id: {id}. Don't use the prefix 'user_' or 'item_'. Just use the id number only, e.g., 1, 2, 3, ..."
                    return observation
                elif analyse_type.lower() not in ['user', 'item']:
                    observation = f"Invalid analyse type: {analyse_type}. It should be either 'user' or 'item'."
                    return observation
                else:
                    try:
                        id = int(id)
                    except (ValueError, TypeError):
                        observation = f"Invalid id: {id}. The id should be an integer."
                        return observation
        
        # Pass task_context to the analysis method
        return self(analyse_type=analyse_type, id=id, task_context=task_context, **kwargs)

if __name__ == '__main__':
    from langchain.prompts import PromptTemplate
    from macrec.utils import init_api, read_prompts
    init_api(read_json('config/api-config.json'))
    prompts = read_prompts('config/prompts/old_system_prompt/react_analyst.json')
    for prompt_name, prompt_template in prompts.items():
        if isinstance(prompt_template, PromptTemplate) and 'task_type' in prompt_template.input_variables:
            prompts[prompt_name] = prompt_template.partial(task_type='rating prediction')
    analyst = Analyst(config_path='config/agents/analyst_ml-100k.json', prompts=prompts)
    user_id, item_id = list(map(int, input('User id and item id: ').split()))
    result = analyst(user_id=user_id, item_id=item_id)
