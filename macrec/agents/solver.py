from typing import Any, Dict, List, Optional
from loguru import logger
from macrec.agents.base import Agent


class Solver(Agent):
    """
    ReWOO Solver Agent: Aggregates worker results to generate final recommendations.
    Takes the planned execution results and synthesizes them into a coherent answer.
    """
    
    def __init__(self, config_path: str = None, config: dict = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        base_message = f"""You are a ReWOO Solver for recommendation tasks. Your role is to aggregate and synthesize worker results into a final recommendation for {task} tasks.

CRITICAL INSTRUCTIONS:
1. You will receive a plan and the execution results from multiple workers
2. Synthesize all the information to provide a coherent, final recommendation
3. Ensure your output matches the expected format for the {task} task
4. Consider all worker outputs and their relationships when forming your solution

Your task is to analyze the worker results and provide the final answer."""
        
        if task == 'sr':
            base_message += """
For Sequential Recommendation:
- CRITICAL: Use ONLY the candidate items provided in the original query - do not generate new items
- The original query contains specific candidate items with IDs - you must rank these exact items
- Provide a ranked list of the candidate item IDs from the original query
- Consider temporal patterns and user preferences from the analysis
- Format: Return only the item IDs in ranked order (e.g., [1311, 627, 71, ...])
- DO NOT create new fictional items or generic recommendations
"""
        elif task == 'rp':
            base_message += """
For Rating Prediction:
- Provide predicted ratings for user-item pairs
- Consider user preferences and item characteristics
- Format: Return numerical rating predictions
"""
        elif task == 'rr':
            base_message += """
For Retrieve & Rank:
- Provide a ranked list of the retrieved candidate items
- Consider analysis results from all candidates
- Format: Return ranked list of item IDs with scores/reasons
"""
        elif task == 'gen':
            base_message += """
For Review Generation:
- Generate a coherent review based on analysis results
- Consider user preferences and item characteristics
- Format: Return a natural language review
"""
            
        return base_message
        
    def user_message(self, plan: str, worker_results: Dict[str, Any], task: str, **kwargs) -> str:
        """Generate user message with plan and worker results."""
        message = f"""Original Plan:
{plan}

Worker Execution Results:
"""
        
        for step_var, result in worker_results.items():
            message += f"{step_var}: {result}\n"
            
        # Include original query data so Solver can see candidate items
        if kwargs.get('data'):
            message += f"""
Original Query Data:
{kwargs['data']}
"""
            
        message += f"""
Based on the above plan and execution results, please provide the final recommendation for this {task.upper()} task."""
        
        return message

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
                # Extract ranked list of items
                import re
                # Look for patterns like [1, 2, 3] or item IDs
                matches = re.findall(r'\[([^\]]+)\]', solution)
                if matches:
                    # Take the first match and parse as list
                    items_str = matches[0].split(',')
                    items = []
                    for item in items_str:
                        item = item.strip()
                        if item.isdigit():
                            items.append(int(item))
                    if items:
                        return items[:10]  # Top 10
                
                # Alternative: look for numbered lists or sequences
                matches = re.findall(r'(?:^|\s)(\d+)(?=\s|$|,)', solution)
                if matches:
                    items = [int(match) for match in matches[:10]]  # Top 10
                    if items:
                        return items
                
                # Fallback: return a default list for testing
                logger.warning(f"Could not extract item list from solution: {solution}")
                return [1, 2, 3, 4, 5]  # Default fallback
                    
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
            return [1, 2, 3, 4, 5]  # Default item list
        elif task == 'rp':
            return 3.0  # Default rating
        else:
            return solution  # Return as-is for other tasks
    
    def get_last_solution(self) -> Optional[str]:
        """Get the most recent solution."""
        if self.solution_history:
            return self.solution_history[-1]['solution']
        return None