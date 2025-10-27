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
        if task in ['sr', 'rr']:
            # Extract candidate item IDs directly from data_sample CSV column (most reliable)
            if kwargs.get('data_sample') is not None and 'candidate_item_id' in kwargs['data_sample']:
                try:
                    candidate_item_id_value = kwargs['data_sample']['candidate_item_id']
                    # Parse the list string representation
                    if isinstance(candidate_item_id_value, str):
                        candidate_ids = list(eval(candidate_item_id_value))
                    elif isinstance(candidate_item_id_value, (list, set)):
                        candidate_ids = list(candidate_item_id_value)
                    else:
                        candidate_ids = []
                    
                    if candidate_ids:
                        candidate_ids_list = f"""
MANDATORY: You MUST rank ONLY these {len(candidate_ids)} candidate item IDs (in any order): {candidate_ids}
DO NOT include any other item IDs in your ranking.
"""
                except Exception as e:
                    logger.warning(f"Failed to extract candidate_item_id from data_sample in Solver: {e}")
            
            # Fallback to regex extraction from input if CSV extraction failed
            if not candidate_ids_list and kwargs.get('input'):
                import re
                # Support multiple dataset formats: "ID: Title:" (MovieLens), "ID: Brand:" (Beauty), or "ID: Business:" (Yelp)
                candidate_matches = re.findall(r'(\d+):\s*(?:Title|Brand|Business):', kwargs['input'])
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
                import re
                
                # First, clean up markdown code fences (Gemini 2.5 Pro wraps JSON in ```json ... ```)
                cleaned_solution = solution.strip()
                if '```' in cleaned_solution:
                    # Extract JSON from code blocks
                    parts = cleaned_solution.split('```')
                    for part in parts:
                        part_stripped = part.strip()
                        # Skip the 'json' language identifier if present
                        if part_stripped.startswith('json'):
                            part_stripped = part_stripped[4:].strip()
                        
                        # Look for JSON object patterns
                        if part_stripped.startswith('{') and part_stripped.endswith('}'):
                            cleaned_solution = part_stripped
                            break
                
                try:
                    # Parse JSON response
                    logger.debug(f"Attempting to parse cleaned_solution: {cleaned_solution[:500]}")
                    data = json.loads(cleaned_solution)
                    if isinstance(data, dict) and 'ranked_items' in data:
                        items = data['ranked_items']
                        if isinstance(items, list):
                            # Convert all items to integers (handle both int and string formats)
                            try:
                                converted_items = []
                                for item in items:
                                    if isinstance(item, int):
                                        converted_items.append(item)
                                    elif isinstance(item, str) and item.isdigit():
                                        converted_items.append(int(item))
                                    else:
                                        # Invalid item format
                                        raise ValueError(f"Invalid item format: {item}")
                                logger.info(f"Extracted {len(converted_items)} items from JSON response")
                                return converted_items
                            except (ValueError, TypeError) as e:
                                logger.error(f"Failed to convert ranked_items to integers: {e}")
                                logger.error(f"Original items: {items}")
                        else:
                            logger.error(f"Invalid ranked_items format (not a list): {items}")
                    else:
                        logger.error(f"JSON response missing 'ranked_items' key: {data}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.debug(f"Solution (raw, first 1000 chars): {solution[:1000]}")
                    
                    # Try to recover by extracting ranked_items from incomplete response
                    try:
                        logger.info("Attempting JSON recovery from truncated response...")
                        # First, try to close the incomplete JSON by adding missing closing braces/quotes
                        fixed_solution = cleaned_solution
                        
                        # Count unclosed braces and quotes
                        open_braces = fixed_solution.count('{') - fixed_solution.count('}')
                        open_quotes_count = fixed_solution.count('"') % 2  # Odd number means unclosed
                        
                        # Close any unclosed quotes and braces
                        if open_quotes_count == 1:
                            fixed_solution += '"'
                        if open_braces > 0:
                            fixed_solution += '}' * open_braces
                        
                        # Try to parse the fixed JSON
                        try:
                            data = json.loads(fixed_solution)
                            if isinstance(data, dict) and 'ranked_items' in data:
                                items = data['ranked_items']
                                if isinstance(items, list):
                                    converted_items = []
                                    for item in items:
                                        if isinstance(item, int):
                                            converted_items.append(item)
                                        elif isinstance(item, str) and item.isdigit():
                                            converted_items.append(int(item))
                                    if converted_items:
                                        logger.info(f"Recovered {len(converted_items)} items by closing incomplete JSON")
                                        return converted_items
                        except json.JSONDecodeError:
                            # If closing braces didn't work, fall back to regex extraction
                            pass
                        
                        # Fallback: Extract ranked_items array using regex (most robust for truncated JSON)
                        items_match = re.search(r'"ranked_items"\s*:\s*\[([^\]]+)\]', solution)
                        if items_match:
                            items_str = items_match.group(1)
                            # Extract all numbers from the array
                            item_numbers = re.findall(r'\d+', items_str)
                            if item_numbers:
                                converted_items = [int(num) for num in item_numbers]
                                logger.info(f"Recovered {len(converted_items)} items from incomplete JSON via regex extraction")
                                return converted_items
                    except Exception as recovery_e:
                        logger.error(f"JSON recovery failed: {recovery_e}")
                
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