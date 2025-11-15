from typing import Any, Dict, List, Optional
from loguru import logger
from macrec.agents.base import Agent


class Planner(Agent):
    """
    ReWOO Planner Agent: Decomposes complex recommendation tasks into sub-problems 
    without making external calls. Creates a structured plan for workers to execute.
    """
    
    def __init__(self, config_path: str = None, config: dict = None, prompt_config: str = None, prompts: dict = None, *args, **kwargs):
        super().__init__(prompts=prompts or {}, prompt_config=prompt_config, *args, **kwargs)
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
        
        # Retriever is always available as a tool
        available_workers.append("Retriever")
        
        # Default workers if system not available
        if not available_workers:
            available_workers = ["Analyst", "Retriever"]
        
        workers_desc = ""
        if "Analyst" in available_workers:
            workers_desc += "- Analyst: Analyzes user preferences, item features, or user-item interactions\n"
        if "Retriever" in available_workers:
            workers_desc += "- Retriever: Retrieves candidate items for a given user (MUST be called as 2nd step after user analysis)\n"
        
        # Load prompt from config (required)
        if 'planner_system_prompt' not in self.prompts:
            raise ValueError("planner_system_prompt not found in prompts config. Please ensure prompt_config is properly loaded.")
            
        base_message = self.prompts['planner_system_prompt'].format(
            task=task,
            available_workers=available_workers,
            workers_desc=workers_desc.rstrip()
        )
        
        # Add task-specific guidance from config
        guidance_key = f'planner_{task}_guidance'
        if guidance_key in self.prompts:
            base_message += self.prompts[guidance_key]
            
        return base_message
        
    def user_message(self, query: str, task: str, **kwargs) -> str:
        """Generate user message for planning with entity extraction."""
        context = ""
        if 'user_id' in kwargs:
            context += f"User ID: {kwargs['user_id']}\n"
        if 'item_id' in kwargs:
            context += f"Item ID: {kwargs['item_id']}\n"
        if 'n_candidate' in kwargs:
            context += f"Number of candidates: {kwargs['n_candidate']}\n"
        if 'history' in kwargs:
            context += f"User history available: Yes\n"
        
        # Add reflection feedback if available
        reflection_context = ""
        if 'reflections' in kwargs and kwargs['reflections'].strip():
            reflection_context = f"\n{kwargs['reflections']}\n"
        
        # For SR/RP/Gen tasks, provide specific planning guidance
        planning_guidance = ""
        if task in ['sr', 'rp', 'gen']:
            # Extract user ID from query
            import re
            
            user_id_match = re.search(r'user[_\s]*id[:\]]*\s*(\d+)', query, re.IGNORECASE)
            
            if user_id_match:
                user_id = user_id_match.group(1)
                
                # NEW WORKFLOW: No candidate items in query, must use Retriever tool
                # Build plan structure with Retriever as 2nd step and placeholder item references
                planning_guidance = f"\nREQUIRED PLAN STRUCTURE:\n"
                planning_guidance += f"#E1 = Analyst[Analyze user {user_id}'s profile, interaction history, and extract preferences]\n"
                planning_guidance += f"#E2 = Retriever[Retrieve candidate items for user {user_id}] (depends on #E1)\n"
                planning_guidance += f"#E3 = Analyst[Analyze 1st candidate item from #E2] (depends on #E2)\n"
                planning_guidance += f"#E4 = Analyst[Analyze 2nd candidate item from #E2] (depends on #E2)\n"
                planning_guidance += f"#E5 = Analyst[Analyze 3rd candidate item from #E2] (depends on #E2)\n"
                planning_guidance += f"... (continue for more candidate items: 4th, 5th, 6th, etc.)\n\n"
                planning_guidance += f"CRITICAL:\n"
                planning_guidance += f"- Step 1 MUST analyze the user\n"
                planning_guidance += f"- Step 2 MUST use Retriever to get candidate items\n"
                planning_guidance += f"- Steps 3+ MUST analyze each candidate using ORDINAL references (1st, 2nd, 3rd, etc.)\n"
                planning_guidance += f"- Do NOT use actual item IDs in the plan - use ordinal positions instead\n"
                planning_guidance += f"- Each item analysis step must depend on #E2\n"
        
        # Use config template (required)
        if 'planner_user_prompt' not in self.prompts:
            raise ValueError("planner_user_prompt not found in prompts config.")
            
        return self.prompts['planner_user_prompt'].format(
            task=task,
            context=context,
            query=query,
            planning_guidance=planning_guidance
        ) + reflection_context

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
            from macrec.llms import OllamaLLM, OpenRouterLLM, GeminiLLM
            if isinstance(self.llm, (OllamaLLM, OpenRouterLLM, GeminiLLM)):
                # For Ollama, OpenRouter, and Gemini, combine system and user messages into single prompt
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
            
            # Handle lines that start with "Plan: #E1 = ..." by removing the "Plan: " prefix
            if line.startswith('Plan: ') and '#E' in line:
                line = line[6:]  # Remove "Plan: " prefix
            
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
                    
                    # Clean up task description by removing dependency annotations
                    # Remove patterns like ", depends_on: #E2" or ", depends on #E2"
                    import re
                    task_desc = re.sub(r',\s*depends[_ ]on:\s*#E\d+', '', task_desc, flags=re.IGNORECASE).strip()
                    
                    # The worker_part should be the worker type
                    worker_type = worker_part
                elif '[' in action:
                    # Handle incomplete/truncated lines that have '[' but no ']'
                    worker_part = action.split('[')[0].strip()
                    worker_type = worker_part
                    task_desc = "Incomplete task description (truncated)"
                    # Skip this incomplete step
                    continue
                    
                # Extract dependencies (from the action part)
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
                elif 'retrieve candidate' in task_desc or 'get candidate' in task_desc:
                    step['worker_type'] = 'Retriever'
                elif 'retrieve' in task_desc or 'candidate' in task_desc:
                    step['worker_type'] = 'Retriever'
                elif 'rank' in task_desc or 'score' in task_desc or 'order' in task_desc:
                    step['worker_type'] = 'Analyst'  # Fallback to Analyst for ranking tasks  
                elif 'interpret' in task_desc or 'generate' in task_desc or 'recommend' in task_desc:
                    step['worker_type'] = 'Analyst'  # Use Analyst as fallback
                else:
                    step['worker_type'] = 'Analyst'  # Default fallback
        
        return steps
    
    def get_last_plan(self) -> Optional[str]:
        """Get the most recent plan."""
        if self.plan_history:
            return self.plan_history[-1]['plan']
        return None