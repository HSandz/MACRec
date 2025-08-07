from typing import Any
from loguru import logger
from langchain.prompts import PromptTemplate

from macrec.agents.base import ToolAgent
from macrec.tools import Wikipedia
from macrec.utils import read_json, parse_action, get_rm

class Searcher(ToolAgent):
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        tool_config: dict[str, dict] = get_rm(config, 'tool_config', {})
        self.get_tools(tool_config)
        self.max_turns = get_rm(config, 'max_turns', 6)
        self.searcher = self.get_LLM(config=config)
        self.json_mode = self.searcher.json_mode
        self.reset()

    @staticmethod
    def required_tools() -> dict[str, type]:
        return {
            'retriever': Wikipedia,
        }

    @property
    def retriever(self) -> Wikipedia:
        return self.tools['retriever']

    @property
    def searcher_prompt(self) -> PromptTemplate:
        if self.json_mode:
            return self.prompts['searcher_prompt_json']
        else:
            return self.prompts['searcher_prompt']

    @property
    def searcher_examples(self) -> str:
        if self.json_mode:
            return self.prompts['searcher_examples_json']
        else:
            return self.prompts['searcher_examples']

    @property
    def hint(self) -> str:
        if 'searcher_hint' not in self.prompts:
            return ''
        return self.prompts['searcher_hint']

    def _build_searcher_prompt(self, **kwargs) -> str:
        # Add command count and repetition warning to the prompt
        command_count = len(self._history)
        remaining_steps = self.max_turns - command_count
        
        # Check for recent repetitive patterns and provide context
        repetition_warning = ""
        if len(self._history) >= 2:
            recent_commands = [turn['command'] for turn in self._history[-2:]]
            if len(set(recent_commands)) == 1:  # All commands are the same
                repetition_warning = f"\nWARNING: You have been repeating the same command '{recent_commands[0]}'. This is wasteful and prohibited. You MUST try a different action or use Finish to complete the search."
        
        # Add explicit command history to show what was done
        command_summary = ""
        if len(self._history) > 0:
            unique_commands = []
            for turn in self._history:
                if turn['command'] not in unique_commands:
                    unique_commands.append(turn['command'])
            command_summary = f"\n\nCOMMANDS ALREADY EXECUTED:\n" + "\n".join([f"- {cmd}" for cmd in unique_commands])
            command_summary += "\n\nDO NOT REPEAT ANY OF THE ABOVE COMMANDS. Choose a different action or use Finish."
        
        # Build the base prompt
        base_prompt = self.searcher_prompt.format(
            examples=self.searcher_examples,
            k=self.retriever.top_k,
            history=self.history,
            max_step=self.max_turns,
            hint=self.hint if len(self._history) + 1 >= self.max_turns else '',
            **kwargs
        )
        
        # Add step counter and repetition warning
        step_info = f"\nYou are at step {command_count + 1}/{self.max_turns}. Remaining steps: {remaining_steps}."
        if remaining_steps <= 2:
            step_info += " You should consider finishing your search soon with the Finish command."
        
        return base_prompt + step_info + repetition_warning + command_summary

    def _prompt_searcher(self, **kwargs) -> str:
        searcher_prompt = self._build_searcher_prompt(**kwargs)
        command = self.searcher(searcher_prompt)
        return command

    def command(self, command: str) -> None:
        logger.debug(f'Command: {command}')
        
        # Enhanced repetition detection similar to analyst
        if len(self._history) >= 2:
            # Check for exact repetition
            recent_commands = [turn['command'] for turn in self._history[-2:]]
            if all(cmd == command for cmd in recent_commands):
                logger.info(f'Detected repetitive command: {command}. Forcing finish.')
                self.finish("Search completed. Detected repetitive pattern, ending search to prevent infinite loop.")
                return
            
            # Check for alternating pattern
            if len(self._history) >= 4:
                last_4_commands = [turn['command'] for turn in self._history[-4:]]
                if last_4_commands[0] == last_4_commands[2] and last_4_commands[1] == last_4_commands[3]:
                    logger.info(f'Detected alternating repetitive commands: {last_4_commands}. Forcing finish.')
                    self.finish("Search completed. Detected alternating repetitive pattern, ending search.")
                    return
        
        log_head = ''
        action_type, argument = parse_action(command, json_mode=self.json_mode)
        if action_type.lower() == 'search':
            observation = self.retriever.search(query=argument)
            log_head = f':violet[Search for] :red[{argument}]:violet[...]\n- '
        elif action_type.lower() == 'lookup':
            if self.json_mode:
                title, term = argument
                observation = self.retriever.lookup(title=title, term=term)
                log_head = f':violet[Lookup for] :red[{term}] :violet[in document] :red[{title}]:violet[...]\n- '
            else:
                try:
                    title, term = argument.split(',')
                    title = title.strip()
                    term = term.strip()
                    observation = self.retriever.lookup(title=title, term=term)
                    log_head = f':violet[Lookup for] :red[{term}] :violet[in document] :red[{title}]:violet[...]\n- '
                except Exception:
                    observation = f'Invalid argument format: {argument}. Must be in the format "title, term".'
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

    def forward(self, requirements: str, *args, **kwargs) -> str:
        while not self.is_finished():
            command = self._prompt_searcher(requirements=requirements)
            self.command(command)
            
            # Break if finished was set during command execution (due to repetition detection)
            if self.finished:
                break
                
        if not self.finished:
            # Force finish if we reached max turns without finishing
            self.finish("Search completed after reaching maximum number of steps.")
            
        return f'Search result: {self.results}'

    def invoke(self, argument: Any, json_mode: bool) -> str:
        if not isinstance(argument, str):
            return f'Invalid argument type: {type(argument)}. Must be a string.'
        return self(requirements=argument)

if __name__ == '__main__':
    from macrec.utils import init_api, read_prompts
    init_api(read_json('config/api-config.json'))
    searcher = Searcher(config_path='config/agents/searcher.json', prompts=read_prompts('config/prompts/agent_prompt/react_search.json'))
    while True:
        requirements = input('Requirements: ')
        print(searcher(requirements=requirements))
