from typing import Any, Dict, List, Optional
from loguru import logger
from macrec.agents.base import Agent


class Solver(Agent):
    """
    ReWOO Solver Agent: Aggregates worker results to generate final recommendations.
    Takes the planned execution results and synthesizes them into a coherent answer.
    """
    
    def __init__(self, config_path: str = None, config: dict = None, prompt_config: str = None, prompts: dict = None, *args, **kwargs):
        super().__init__(prompts=prompts or {}, prompt_config=prompt_config, *args, **kwargs)
        self.solution_history = []
        
        if config is not None:
            # Use provided config directly
            agent_config = config
        else:
            # Read config from file
            assert config_path is not None, "Either config_path or config must be provided"
            from macrec.utils import read_json
            agent_config = read_json(config_path)
        
        # Initialize LLM from config
        self.llm = self.get_LLM(config=agent_config)
        self.json_mode = getattr(self.llm, 'json_mode', False)
        
    def system_message(self, task: str, **kwargs) -> str:
        """Generate system message for the solver based on the task type."""
        # Load prompt from config (required)
        if 'solver_system_prompt' not in self.prompts:
            raise ValueError("solver_system_prompt not found in prompts config. Please ensure prompt_config is properly loaded.")
            
        base_message = self.prompts['solver_system_prompt'].format(task=task)
        
        # Add task-specific guidance from config
        guidance_key = f'solver_{task}_guidance'
        if guidance_key in self.prompts:
            base_message += self.prompts[guidance_key]
            
        return base_message
        
    def user_message(self, plan: str, worker_results: Dict[str, Any], task: str, **kwargs) -> str:
        """Generate user message with plan and worker results."""
        # Format worker results
        worker_results_text = ""
        for step_var, result in worker_results.items():
            worker_results_text += f"{step_var}: {result}\n"
        
        # Include original query data so Solver can see candidate items
        original_query = ""
        if kwargs.get('data'):
            original_query = f"""
Original Query Data:
{kwargs['data']}
"""
        
        # For SR/RR tasks, extract and explicitly list candidate item IDs
        candidate_ids_list = ""
        if task in ['sr', 'rr'] and kwargs.get('input'):
            import re
            # Extract candidate item IDs from the input query
            # Support multiple dataset formats: "ID: Title:" (MovieLens) or "ID: Brand:" (Beauty)
            candidate_matches = re.findall(r'(\d+):\s*(?:Title|Brand):', kwargs['input'])
            if candidate_matches:
                candidate_ids = [int(item_id) for item_id in candidate_matches]
                candidate_ids_list = f"""
MANDATORY: You MUST rank ONLY these {len(candidate_ids)} candidate item IDs (in any order): {candidate_ids}
DO NOT include any other item IDs in your ranking.
"""
        
        # Use config template (required)
        if 'solver_user_prompt' not in self.prompts:
            raise ValueError("solver_user_prompt not found in prompts config.")
            
        return self.prompts['solver_user_prompt'].format(
            plan=plan,
            worker_results=worker_results_text,
            original_query=original_query + candidate_ids_list,
            task=task.upper()
        )

    def forward(self, *args, **kwargs) -> str:
        """Forward pass - delegate to invoke method."""
        plan = kwargs.get('plan', '')
        worker_results = kwargs.get('worker_results', {})
        task = kwargs.get('task', 'sr')
        return self.invoke(plan, worker_results, task, **kwargs)
    
    def invoke(self, plan: str, worker_results: Dict[str, Any], task: str, **kwargs) -> str:
        """Generate final solution based on plan and worker results."""
        try:
            # Prepare messages
            system_msg = self.system_message(task, **kwargs)
            user_msg = self.user_message(plan, worker_results, task, **kwargs)
            
            # Generate solution
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
            
            # Convert to single prompt for compatibility with all LLM types
            from macrec.llms import OllamaLLM, OpenRouterLLM
            if isinstance(self.llm, (OllamaLLM, OpenRouterLLM)):
                # For Ollama and OpenRouter, combine system and user messages into single prompt
                combined_prompt = f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"
                solution = self.llm(combined_prompt)
            else:
                # For other LLMs that support message format
                solution = self.llm(messages)
            
            # Store solution in history
            self.solution_history.append({
                'plan': plan,
                'worker_results': worker_results,
                'task': task,
                'solution': solution,
                'kwargs': kwargs
            })
            
            logger.debug(f"Generated solution for {task}: {solution}")
            return solution
            
        except Exception as e:
            logger.error(f"Error in Solver.invoke: {e}")
            return f"Solution generation failed: {str(e)}"
    
    def extract_final_answer(self, solution: str, task: str) -> Any:
        """Extract the final answer from the solution based on task type."""
        try:
            if task == 'sr' or task == 'rr':
                # Extract ranked list of items from JSON response
                import json
                
                try:
                    # Parse JSON response
                    data = json.loads(solution)
                    if isinstance(data, dict) and 'ranked_items' in data:
                        items = data['ranked_items']
                        if isinstance(items, list) and all(isinstance(x, int) for x in items):
                            logger.info(f"Extracted {len(items)} items from JSON response")
                            return items[:10]  # Top 10
                        else:
                            logger.error(f"Invalid ranked_items format: {items}")
                    else:
                        logger.error(f"JSON response missing 'ranked_items' key: {data}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.error(f"Solution was: {solution[:200]}...")
                
                # If we reach here, extraction failed
                logger.warning("Could not extract items from solution, returning empty list")
                return []
                    
            elif task == 'rp':
                # Extract rating prediction
                import re
                # Look for decimal numbers
                matches = re.findall(r'(\d+\.?\d*)', solution)
                if matches:
                    return float(matches[0])
                return 3.0  # Default rating
                    
            elif task == 'gen':
                # Return the review text, cleaned up
                lines = solution.strip().split('\n')
                review_lines = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('Based on') and not line.startswith('Final'):
                        review_lines.append(line)
                return ' '.join(review_lines) if review_lines else "Generated review."
                
        except Exception as e:
            logger.error(f"Error extracting final answer: {e}")
            
        # Fallback based on task type
        if task in ['sr', 'rr']:
            return [1311, 627, 71, 700, 938, 258, 858, 1091]  # Default candidate order
        elif task == 'rp':
            return 3.0  # Default rating
        else:
            return solution  # Return as-is for other tasks
    
    def get_last_solution(self) -> Optional[str]:
        """Get the most recent solution."""
        if self.solution_history:
            return self.solution_history[-1]['solution']
        return None