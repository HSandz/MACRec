from typing import Any, Dict, List, Optional
from loguru import logger
from macrec.agents.base import Agent


class Planner(Agent):
    """
    ReWOO Planner Agent: Decomposes complex recommendation tasks into sub-problems 
    without making external calls. Creates a structured plan for workers to execute.
    """
    
    def __init__(self, config_path: str = None, config: dict = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plan_history = []
        
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
        """Generate system message for the planner based on the task type."""
        # Check available workers from the system
        available_workers = []
        if hasattr(self, 'system') and self.system:
            if hasattr(self.system, 'analyst') and self.system.analyst:
                available_workers.append("Analyst")
            if hasattr(self.system, 'retriever') and self.system.retriever:
                available_workers.append("Retriever") 
            if hasattr(self.system, 'searcher') and self.system.searcher:
                available_workers.append("Searcher")
            if hasattr(self.system, 'interpreter') and self.system.interpreter:
                available_workers.append("Interpreter")
        
        # Default workers if system not available
        if not available_workers:
            available_workers = ["Analyst", "Retriever", "Searcher", "Interpreter"]
        
        workers_desc = ""
        if "Analyst" in available_workers:
            workers_desc += "- Analyst: Analyzes user preferences, item features, or user-item interactions\n"
        if "Retriever" in available_workers:
            workers_desc += "- Retriever: Retrieves candidate items or similar users/items\n"
        if "Searcher" in available_workers:
            workers_desc += "- Searcher: Searches for relevant information in knowledge bases\n"
        if "Interpreter" in available_workers:
            workers_desc += "- Interpreter: Interprets natural language queries or requirements\n"
        
        base_message = f"""You are a ReWOO Planner for recommendation tasks. Your role is to decompose complex {task} tasks into a structured plan of sub-problems without making any external calls.

CRITICAL INSTRUCTIONS:
1. You must create a step-by-step plan that can be executed by workers
2. Do NOT execute any actions yourself - only plan them
3. Use variables like #E1, #E2, etc. to reference results from previous steps
4. Each step should specify the worker type and the exact task
5. ONLY use workers that are actually available: {available_workers}

Available Workers:
{workers_desc.rstrip()}

Plan Format:
Plan: #E1 = WorkerType[task_description]
#E2 = WorkerType[task_description, depends_on: #E1]
#E3 = WorkerType[task_description, depends_on: #E1, #E2]

IMPORTANT: If only Analyst is available, create a simple plan that analyzes the user, then analyzes candidate items, then provides a ranking based on the analysis.
"""
        
        if task == 'sr':
            base_message += """
For Sequential Recommendation:
- Plan should include analyzing user history, identifying patterns, retrieving candidates, and ranking
- Consider temporal aspects and sequence patterns
"""
        elif task == 'rp':
            base_message += """
For Rating Prediction:
- Plan should include analyzing user preferences, item characteristics, and predicting ratings
- Consider user-item interaction patterns
"""
        elif task == 'rr':
            base_message += """
For Retrieve & Rank:
- Plan should include retrieving candidate items, analyzing each candidate, and ranking them
- Must analyze all retrieved candidates before ranking
"""
        elif task == 'gen':
            base_message += """
For Review Generation:
- Plan should include analyzing item features, user preferences, and generating coherent reviews
- Consider review style and content requirements
"""
            
        return base_message
        
    def user_message(self, query: str, task: str, **kwargs) -> str:
        """Generate user message for planning."""
        context = ""
        if 'user_id' in kwargs:
            context += f"User ID: {kwargs['user_id']}\n"
        if 'item_id' in kwargs:
            context += f"Item ID: {kwargs['item_id']}\n"
        if 'n_candidate' in kwargs:
            context += f"Number of candidates: {kwargs['n_candidate']}\n"
        if 'history' in kwargs:
            context += f"User history available: Yes\n"
            
        return f"""Task: {task.upper()}
{context}
Query: {query}

Please create a detailed execution plan for this {task} task. Break it down into specific steps that workers can execute independently."""

    def forward(self, *args, **kwargs) -> str:
        """Forward pass - delegate to invoke method."""
        query = kwargs.get('query', '')
        task = kwargs.get('task', 'sr')
        return self.invoke(query, task, **kwargs)
    
    def invoke(self, query: str, task: str, **kwargs) -> str:
        """Create a plan for the given recommendation task."""
        try:
            # Prepare messages
            system_msg = self.system_message(task, **kwargs)
            user_msg = self.user_message(query, task, **kwargs)
            
            # Generate plan
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
            
            # Convert to single prompt for compatibility with all LLM types
            from macrec.llms import OllamaLLM, OpenRouterLLM
            if isinstance(self.llm, (OllamaLLM, OpenRouterLLM)):
                # For Ollama and OpenRouter, combine system and user messages into single prompt
                combined_prompt = f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"
                plan = self.llm(combined_prompt)
            else:
                # For other LLMs that support message format
                plan = self.llm(messages)
            
            # Store plan in history
            self.plan_history.append({
                'query': query,
                'task': task,
                'plan': plan,
                'kwargs': kwargs
            })
            
            logger.debug(f"Generated plan for {task}: {plan}")
            return plan
            
        except Exception as e:
            logger.error(f"Error in Planner.invoke: {e}")
            return f"Planning failed: {str(e)}"
    
    def parse_plan(self, plan: str) -> List[Dict[str, Any]]:
        """Parse the generated plan into structured steps."""
        steps = []
        lines = plan.split('\n')
        
        current_step = None
        worker_type_mapping = {}
        
        for line in lines:
            line = line.strip()
            
            # Parse basic plan format: #E1 = Worker[task]
            if '=' in line and line.startswith('#E'):
                # Extract step variable and action
                var_part, action_part = line.split('=', 1)
                variable = var_part.strip()
                action = action_part.strip()
                
                # Extract worker type from the format: WorkerType[task_description]
                worker_type = 'Unknown'
                task_desc = "No description"
                
                if '[' in action and ']' in action:
                    # Split at the first '[' to separate worker type and task
                    worker_part = action.split('[')[0].strip()
                    task_desc = action.split('[')[1].split(']')[0]
                    
                    # The worker_part should be the worker type
                    worker_type = worker_part
                    
                # Extract dependencies (from the task description)
                dependencies = []
                if 'depends_on:' in action:
                    dep_part = action.split('depends_on:')[1]
                    for dep in dep_part.split(','):
                        dep = dep.strip().replace(']', '').replace(')', '')
                        if dep.startswith('#E'):
                            dependencies.append(dep)
                
                current_step = {
                    'variable': variable,
                    'worker_type': worker_type,
                    'task_description': task_desc,
                    'dependencies': dependencies,
                    'raw_action': action
                }
                steps.append(current_step)
            
            # Parse detailed worker type specifications: "#E1 = Analyze User History" followed by "Worker Type: Analyst"
            elif line.startswith('#E') and '=' in line and 'Worker Type:' not in line:
                # This might be a step title, continue to look for worker type
                continue
            elif 'Worker Type:' in line and current_step:
                # Extract the actual worker type
                worker_type = line.split('Worker Type:')[1].strip()
                current_step['worker_type'] = worker_type
                # Map the variable to worker type for dependency resolution
                worker_type_mapping[current_step['variable']] = worker_type
            elif 'Task:' in line and current_step:
                # Extract more detailed task description
                task_desc = line.split('Task:')[1].strip()
                current_step['task_description'] = task_desc
        
        # If we still have Unknown worker types, try to infer them from task descriptions
        for step in steps:
            if step['worker_type'] == 'Unknown':
                task_desc = step['task_description'].lower()
                if 'analyz' in task_desc or 'examine' in task_desc or 'pattern' in task_desc:
                    step['worker_type'] = 'Analyst'
                elif 'retrieve' in task_desc or 'search' in task_desc or 'candidate' in task_desc:
                    step['worker_type'] = 'Retriever'
                elif 'rank' in task_desc or 'score' in task_desc or 'order' in task_desc:
                    step['worker_type'] = 'Searcher'  
                elif 'interpret' in task_desc or 'generate' in task_desc or 'recommend' in task_desc:
                    step['worker_type'] = 'Interpreter'
                else:
                    step['worker_type'] = 'Analyst'  # Default fallback
        
        return steps
    
    def get_last_plan(self) -> Optional[str]:
        """Get the most recent plan."""
        if self.plan_history:
            return self.plan_history[-1]['plan']
        return None