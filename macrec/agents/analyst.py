from typing import Any
from loguru import logger

from macrec.agents.base import ToolAgent
from macrec.tools import InfoDatabase, InteractionRetriever
from macrec.utils import read_json, get_rm, parse_action

class Analyst(ToolAgent):
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        tool_config: dict[str, dict] = get_rm(config, 'tool_config', {})
        self.get_tools(tool_config)
        self.max_turns = get_rm(config, 'max_turns', 15)  # Reduced default from 20 to 15
        self.analyst = self.get_LLM(config=config)
        self.json_mode = self.analyst.json_mode
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
        
        # Check for recent repetitive patterns
        repetition_warning = ""
        if len(self._history) >= 2:
            recent_commands = [turn['command'] for turn in self._history[-2:]]
            if len(set(recent_commands)) == 1:  # All commands are the same
                repetition_warning = f"\nWARNING: You have been repeating the same command '{recent_commands[0]}'. Please try a different action or use Finish to complete the analysis."
        
        prompt = self.analyst_prompt.format(
            examples=self.analyst_examples,
            fewshot=self.analyst_fewshot,
            history=self.history,
            max_step=self.max_turns,
            hint=self.hint if len(self._history) + 1 >= self.max_turns else '',
            **kwargs
        )
        
        # Add step counter and repetition warning
        step_info = f"\nYou are at step {command_count + 1}/{self.max_turns}. Remaining steps: {remaining_steps}."
        if remaining_steps <= 3:
            step_info += " You should consider finishing your analysis soon."
        
        return prompt + step_info + repetition_warning

    def _prompt_analyst(self, **kwargs) -> str:
        analyst_prompt = self._build_analyst_prompt(**kwargs)
        command = self.analyst(analyst_prompt)
        return command

    def command(self, command: str) -> None:
        logger.debug(f'Command: {command}')
        
        # Check for repetitive commands to prevent infinite loops
        if len(self._history) >= 3:
            recent_commands = [turn['command'] for turn in self._history[-3:]]
            if all(cmd == command for cmd in recent_commands):
                logger.warning(f'Detected repetitive command: {command}. Forcing finish.')
                # Force finish with a basic analysis result
                self.finish(f"Analysis completed for {command}. Detected repetitive pattern, ending analysis.")
                return
        
        log_head = ''
        action_type, argument = parse_action(command, json_mode=self.json_mode)
        if action_type.lower() == 'userinfo':
            try:
                query_user_id = int(argument)
                observation = self.info_retriever.user_info(user_id=query_user_id)
                log_head = f':violet[Look up UserInfo of user] :red[{query_user_id}]:violet[...]\n- '
            except ValueError or TypeError:
                observation = f"Invalid user id: {argument}"
        elif action_type.lower() == 'iteminfo':
            try:
                query_item_id = int(argument)
                observation = self.info_retriever.item_info(item_id=query_item_id)
                log_head = f':violet[Look up ItemInfo of item] :red[{query_item_id}]:violet[...]\n- '
            except ValueError or TypeError:
                observation = f"Invalid item id: {argument}"
        elif action_type.lower() == 'userhistory':
            valid = True
            if self.json_mode:
                if not isinstance(argument, list) or len(argument) != 2:
                    observation = f"Invalid user id and retrieval number: {argument}"
                    valid = False
                else:
                    query_user_id, k = argument
                    if not isinstance(query_user_id, int) or not isinstance(k, int):
                        observation = f"Invalid user id and retrieval number: {argument}"
                        valid = False
            else:
                try:
                    query_user_id, k = argument.split(',')
                    query_user_id = int(query_user_id)
                    k = int(k)
                except ValueError or TypeError:
                    observation = f"Invalid user id and retrieval number: {argument}"
                    valid = False
            if valid:
                observation = self.interaction_retriever.user_retrieve(user_id=query_user_id, k=k)
                log_head = f':violet[Look up UserHistory of user] :red[{query_user_id}] :violet[with at most] :red[{k}] :violet[items...]\n- '
        elif action_type.lower() == 'itemhistory':
            valid = True
            if self.json_mode:
                if not isinstance(argument, list) or len(argument) != 2:
                    observation = f"Invalid item id and retrieval number: {argument}"
                    valid = False
                else:
                    query_item_id, k = argument
                    if not isinstance(query_item_id, int) or not isinstance(k, int):
                        observation = f"Invalid item id and retrieval number: {argument}"
                        valid = False
            else:
                try:
                    query_item_id, k = argument.split(',')
                    query_item_id = int(query_item_id)
                    k = int(k)
                except ValueError or TypeError:
                    observation = f"Invalid item id and retrieval number: {argument}"
                    valid = False
            if valid:
                observation = self.interaction_retriever.item_retrieve(item_id=query_item_id, k=k)
                log_head = f':violet[Look up ItemHistory of item] :red[{query_item_id}] :violet[with at most] :red[{k}] :violet[users...]\n- '
        elif action_type.lower() == 'finish':
            observation = self.finish(results=argument)
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

    def forward(self, id: int, analyse_type: str, *args: Any, **kwargs: Any) -> str:
        assert self.system.data_sample is not None, "Data sample is not provided."
        assert 'user_id' in self.system.data_sample, "User id is not provided."
        assert 'item_id' in self.system.data_sample, "Item id is not provided."
        self.interaction_retriever.reset(user_id=self.system.data_sample['user_id'], item_id=self.system.data_sample['item_id'])
        while not self.is_finished():
            command = self._prompt_analyst(id=id, analyse_type=analyse_type)
            self.command(command)
        if not self.finished:
            return "Analyst did not return any result."
        return self.results

    def invoke(self, argument: Any, json_mode: bool) -> str:
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
                    except ValueError or TypeError:
                        observation = f"Invalid id: {id}. The id should be an integer."
                        return observation
        return self(analyse_type=analyse_type, id=id)

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
