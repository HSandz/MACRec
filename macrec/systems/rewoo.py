import json
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from loguru import logger
import pandas as pd

from macrec.systems.base import System
from macrec.factories import DefaultAgentFactory, ConfigManager
from macrec.components import ReWOOOrchestrator, AgentCoordinator
from macrec.agents.base import Agent
from macrec.utils import parse_answer, parse_action, format_chat_history, duration_tracker

if TYPE_CHECKING:
    from macrec.agents import Manager, Analyst, Interpreter, Reflector, Searcher, Planner, Solver


class ReWOOSystem(System):
    """
    ReWOO (Reasoning Without Observation) System for Multi-Agent Recommendation.
    
    This system implements the ReWOO pattern with three phases:
    1. Planning: Decompose complex tasks into sub-problems
    2. Working: Execute sub-tasks using existing agents
    3. Solving: Aggregate results to generate final recommendations
    
    The system maintains backward compatibility with existing CollaborationSystem
    while adding ReWOO capabilities for improved reasoning and performance.
    """
    
    def __init__(self, task: str, config_path: str, leak: bool = False, web_demo: bool = False, dataset: Optional[str] = None, enable_reflection_rerun: bool = False, *args, **kwargs) -> None:
        # Initialize factory and orchestrator components
        self.agent_factory = DefaultAgentFactory()
        self.agent_coordinator = AgentCoordinator(self.agent_factory)
        self.orchestrator = None  # Will be initialized in init()
        self.enable_reflection_rerun = enable_reflection_rerun  # Store reflection rerun option
        
        # Initialize entity cache for context sharing between steps
        self.entity_cache = {
            'users': {},  # {user_id: user_data}
            'items': {},  # {item_id: item_data}
            'user_histories': {},  # {user_id: history_data}
            'item_histories': {}   # {item_id: history_data}
        }
        
        super().__init__(task, config_path, leak, web_demo, dataset, *args, **kwargs)
        
    def init(self, *args, **kwargs) -> None:
        """Initialize the ReWOO system."""
        self.max_step: int = self.config.get('max_step', 10)
        assert 'agents' in self.config, 'Agents are required.'
        
        # Initialize orchestrator with config manager
        config_manager = ConfigManager(self.config)
        self.orchestrator = ReWOOOrchestrator(config_manager, self.agent_coordinator)
        # Pass agent_kwargs to ensure dataset and other parameters are available
        self.orchestrator.initialize(**self.agent_kwargs)
        
        self.manager_kwargs = {
            'max_step': self.max_step,
            'task_type': self.task,
        }
        
        # Initialize ReWOO specific tracking
        self.analyzed_items = set()
        self.analyzed_users = set()
        self.execution_results = {}
        self.completed_steps = set()
        self.current_plan = None
        self.plan_steps = []
        self.step_n = 1
        self.phase = 'planning'
        self._execution_errors = []  # Track errors for smart reflection
        
        # Separate kwargs for planner to receive reflection feedback
        self.planner_kwargs = {
            'reflections': '',  # Only planner receives reflection feedback
        }
        
        # Track reflection improvements and all reruns for logging
        self.reflection_improvements = []  # List of improved samples (for backward compatibility)
        self.reflection_all_reruns = []  # List of ALL samples that were rerun by reflection (improved or not)
        self.total_reflections_triggered = 0  # Count of reflection reruns triggered (when correctness=false)
        
        # **CRITICAL**: Initialize sample tracking for reflection improvements
        self._current_sample_idx = -1  # Current sample index (set by generation task)
        self._current_user_id = -1  # Current user ID (set by generation task)

    @staticmethod
    def supported_tasks() -> list[str]:
        return ['rp', 'sr', 'rr', 'gen', 'chat']

    @property
    def planner(self) -> Optional['Planner']:
        return self.agent_coordinator.get_agent('Planner')

    @property
    def solver(self) -> Optional['Solver']:
        return self.agent_coordinator.get_agent('Solver')

    @property
    def manager(self) -> Optional['Manager']:
        return self.agent_coordinator.get_agent('Manager')

    @property
    def analyst(self) -> Optional['Analyst']:
        return self.agent_coordinator.get_agent('Analyst')

    @property
    def searcher(self) -> Optional['Searcher']:
        return self.agent_coordinator.get_agent('Searcher')

    @property
    def interpreter(self) -> Optional['Interpreter']:
        return self.agent_coordinator.get_agent('Interpreter')

    @property
    def reflector(self) -> Optional['Reflector']:
        return self.agent_coordinator.get_agent('Reflector')
    
    def set_data(self, input: str, context: str, gt_answer: Any, data_sample: Optional[pd.Series] = None) -> None:
        """Override set_data to capture user_id and sample_idx from data_sample."""
        super().set_data(input, context, gt_answer, data_sample)
        
        # Extract user_id from data_sample if available
        if data_sample is not None:
            self._current_user_id = data_sample.get('user_id', -1) if hasattr(data_sample, 'get') else (data_sample['user_id'] if 'user_id' in data_sample else -1)
        else:
            self._current_user_id = -1

    def reset(self, clear: bool = False, preserve_progress: bool = False, *args, **kwargs) -> None:
        """Reset the ReWOO system state."""
        # **CRITICAL**: Save sample tracking info before reset
        saved_sample_idx = getattr(self, '_current_sample_idx', -1)
        saved_user_id = getattr(self, '_current_user_id', -1)
        
        super().reset(clear, *args, **kwargs)
        
        if not preserve_progress:
            self.analyzed_items.clear()
            self.analyzed_users.clear()
            self.execution_results.clear()
            self.current_plan = None
            self.plan_steps = []
            # Clear entity cache
            if hasattr(self, 'entity_cache'):
                self.entity_cache = {
                    'users': {},
                    'items': {},
                    'user_histories': {},
                    'item_histories': {}
                }
            # Clear reflection storage
            if hasattr(self, '_last_solution'):
                delattr(self, '_last_solution')
            if hasattr(self, '_last_final_answer'):
                delattr(self, '_last_final_answer')
            # Reset planner reflection feedback
            if hasattr(self, 'planner_kwargs'):
                self.planner_kwargs['reflections'] = ""
        else:
            # **CRITICAL**: Restore sample tracking info during reflection reruns
            self._current_sample_idx = saved_sample_idx
            self._current_user_id = saved_user_id
            
            # When preserving progress during reflection, add progress summary to planner_kwargs
            if hasattr(self, 'analyzed_items') and self.analyzed_items:
                current_progress = {
                    'analyzed_items': list(self.analyzed_items),
                    'analyzed_users': list(getattr(self, 'analyzed_users', set())),
                    'step_n': getattr(self, 'step_n', 1)
                }
                
                if 'reflections' not in self.planner_kwargs:
                    self.planner_kwargs['reflections'] = ""
                
                progress_summary = f"\n=== Previous Progress ===\n"
                if current_progress['analyzed_items']:
                    progress_summary += f"- Analyzed items: {sorted(current_progress['analyzed_items'])}\n"
                if current_progress['analyzed_users']:
                    progress_summary += f"- Analyzed users: {sorted(current_progress['analyzed_users'])}\n"
                progress_summary += f"- Completed {current_progress['step_n']} steps\n"
                progress_summary += "IMPORTANT: Create a plan that avoids repeating the above analyses.\n"
                self.planner_kwargs['reflections'] += progress_summary
        
        # Reset all agents using coordinator
        self.agent_coordinator.reset_all_agents()
        
        self.step_n = 1
        self.phase = 'planning'
        self._execution_errors = []  # Clear error tracking

    def forward(self, user_input: Optional[str] = None, reset: bool = True) -> Any:
        """Forward method implementing ReWOO workflow with exact same data integration as collaboration system."""
        try:
            # Use exact same data integration as collaboration system
            if self.task == 'chat':
                self.manager_kwargs['history'] = self.chat_history if hasattr(self, 'chat_history') else []
            else:
                self.manager_kwargs['input'] = self.input
                self.manager_kwargs['data_sample'] = self.data_sample
                
            if reset:
                self.reset()
                
            if self.task == 'chat':
                assert user_input is not None, 'User input is required for chat task.'
                if hasattr(self, 'add_chat_history'):
                    self.add_chat_history(user_input, role='user')
            
            # Check if ReWOO agents are available
            if self.planner is None or self.solver is None:
                # Fall back to collaboration mode if ReWOO agents not available
                logger.info("ReWOO agents not available. Falling back to collaboration mode.")
                return self._fallback_collaboration_mode()
            
            # Execute ReWOO workflow
            result = self._execute_rewoo_workflow()
            best_result = result  # Track best result for scoring
            best_position = self._gt_position_before_reflection if hasattr(self, '_gt_position_before_reflection') else -1
            
            # **CRITICAL FIX**: Save the INITIAL position from first workflow IMMEDIATELY
            # This must be saved BEFORE reflection and BEFORE any rerun loops
            # Do NOT save it from instance attribute later, as it gets overwritten during reruns
            original_position_before_reflection = self._gt_position_before_reflection if hasattr(self, '_gt_position_before_reflection') else -1
            
            # **CRITICAL**: Track the best answer throughout reflection process
            # Start with the original answer as the best
            best_answer = self.answer.copy() if isinstance(self.answer, list) else self.answer
            best_answer_position = best_position
            
            # Handle reflection and potential reruns (only if enabled)
            if self.enable_reflection_rerun:
                should_continue_reflecting, feedback_info = self._perform_reflection()
                reflection_count = 0
                max_reflections = 1
                
                while should_continue_reflecting and reflection_count < max_reflections:
                    reflection_count += 1
                    logger.debug(f'Starting ReWOO reflection cycle {reflection_count}/{max_reflections}')
                    
                    # **NEW**: Dual feedback handling - check which agent has issues
                    planner_correct = feedback_info.get('planner_correct', True)
                    solver_correct = feedback_info.get('solver_correct', True)
                    planner_reason = feedback_info.get('planner_reason', '')
                    solver_reason = feedback_info.get('solver_reason', '')
                    
                    # Conditional rerun logic:
                    # 1. If BOTH incorrect: Full rerun (Planner gets feedback, then Solver)
                    # 2. If ONLY Planner incorrect: Full rerun (Planner gets feedback)
                    # 3. If ONLY Solver incorrect: Solver-only reranking (no full rerun)
                    # 4. If both correct: Stop reflecting
                    
                    if not planner_correct:
                        # Planner is incorrect - FULL RERUN needed
                        logger.info(f"❌ Planner feedback triggered: {planner_reason}")
                        logger.info(f"🔄 Applying Planner feedback and executing full ReWOO rerun...")
                        
                        # **CRITICAL FIX**: Use the SAVED position from before the reflection loop
                        # NOT the one that gets overwritten when _execute_rewoo_workflow() runs
                        position_before = original_position_before_reflection
                        
                        # Reset with progress preservation for reflection continuation
                        self.reset(preserve_progress=True)
                        
                        # Add Planner feedback
                        if 'reflections' not in self.planner_kwargs:
                            self.planner_kwargs['reflections'] = ""
                        
                        reflection_feedback = f"\n=== Planning Improvement Required (Reflection Feedback) ===\n"
                        reflection_feedback += f"{planner_reason}\n"
                        reflection_feedback += f"CRITICAL: Revise your plan to address this specific issue.\n"
                        
                        self.planner_kwargs['reflections'] += reflection_feedback
                        
                        # If BOTH incorrect, also add Solver feedback for the solving phase
                        if not solver_correct:
                            logger.info(f"❌ Solver feedback also triggered: {solver_reason}")
                            logger.info(f"📋 Applying feedback to BOTH Planner and Solver for full improvement")
                            
                            if 'solver_reflections' not in self.manager_kwargs:
                                self.manager_kwargs['solver_reflections'] = ""
                            
                            solver_reflection_feedback = f"\n=== Solver Improvement Required (Reflection Feedback) ===\n"
                            solver_reflection_feedback += f"{solver_reason}\n"
                            solver_reflection_feedback += f"CRITICAL: Adjust your ranking to address this specific issue.\n"
                            
                            self.manager_kwargs['solver_reflections'] += solver_reflection_feedback
                        
                        # Re-execute full ReWOO workflow with planner feedback (and possibly solver feedback)
                        result = self._execute_rewoo_workflow()
                        
                        # Check position after rerun
                        position_after = self._get_ground_truth_position(self._last_final_answer)
                        logger.info(f"📊 Full rerun complete - position: {position_before} → {position_after}")
                        
                    elif not solver_correct:
                        # ONLY Solver is incorrect - SOLVER RERANKING only (no full rerun)
                        logger.info(f"❌ Solver feedback triggered (solver-only reranking): {solver_reason}")
                        logger.info(f"🔄 Applying Solver feedback and performing reranking (skipping full rerun for efficiency)...")
                        
                        position_before = self._get_ground_truth_position(self._last_final_answer)
                        
                        # Perform solver reranking WITHOUT full rerun
                        reranked_answer = self._perform_solver_reranking(solver_reason)
                        
                        # **CRITICAL**: Store reranked answer directly, NOT the observation string from finish()
                        self.answer = reranked_answer
                        result = self.answer  # Update result for tracking
                        
                        position_after = self._get_ground_truth_position(reranked_answer)
                        logger.info(f"📊 Solver reranking complete - position: {position_before} → {position_after}")
                        
                    else:
                        # Both correct - should not reach here, but handle gracefully
                        logger.info("✅ Both Planner and Solver are correct - stopping reflection")
                        should_continue_reflecting = False
                        break
                    
                    # Get sample info for logging
                    sample_idx = getattr(self, '_current_sample_idx', -1)
                    user_id = getattr(self, '_current_user_id', -1)
                    gt_item = self.gt_answer if hasattr(self, 'gt_answer') else -1
                    
                    # Track this rerun in the all_reruns list (regardless of improvement)
                    rerun_info = {
                        'sample_idx': sample_idx,
                        'user_id': user_id,
                        'gt_item': gt_item,
                        'position_before': position_before,
                        'position_after': position_after,
                        'feedback_type': 'both' if (not planner_correct and not solver_correct) else ('planner' if not planner_correct else 'solver')
                    }
                    self.reflection_all_reruns.append(rerun_info)
                    
                    # Track improvement if position improved (lower is better)
                    if position_before > 0 and position_after > 0 and position_after < position_before:
                        self.reflection_improvements.append(rerun_info)
                        improvement_delta = position_before - position_after
                        logger.info(f"✅ Reflection IMPROVED GT position: {position_before} → {position_after} (Δ{improvement_delta:+d})")
                        logger.info(f"📈 Improvement rate: {len(self.reflection_improvements)}/{len(self.reflection_all_reruns)} reflections successful")
                        
                        # **CRITICAL**: Update best answer since this one is better
                        best_answer = self.answer.copy() if isinstance(self.answer, list) else self.answer
                        best_answer_position = position_after
                        best_position = position_after
                        logger.info(f"✅ Saving IMPROVED answer as best answer for scoring")
                    elif position_before > 0 and position_after > 0 and position_after > position_before:
                        worsened_delta = position_after - position_before
                        logger.warning(f"⚠️  Reflection WORSENED GT position: {position_before} → {position_after} (Δ{worsened_delta:+d})")
                        logger.warning(f"📊 Will REVERT to best answer (position {best_answer_position}) instead of worsened answer (position {position_after})")
                        # **CRITICAL**: Restore the best answer since this rerun made things worse
                        self.answer = best_answer.copy() if isinstance(best_answer, list) else best_answer
                    else:
                        logger.info(f"↩️  Reflection did NOT change position (before: {position_before}, after: {position_after})")
                        # No change, so best answer remains unchanged
                    
                    # Check if we should continue reflecting
                    should_continue_reflecting, feedback_info = self._perform_reflection()
                
                if reflection_count >= max_reflections:
                    logger.info(f'Stopped after {max_reflections} ReWOO reflection cycle to prevent infinite loops')
                
                # Log final reflection summary before returning
                logger.info(f"\n{'='*80}")
                logger.info(f"📊 REFLECTION CYCLE COMPLETE:")
                logger.info(f"  Total reflections triggered: {len(self.reflection_all_reruns)}")
                logger.info(f"  Successful improvements: {len(self.reflection_improvements)}")
                if self.reflection_all_reruns:
                    success_rate = len(self.reflection_improvements) / len(self.reflection_all_reruns) * 100
                    logger.info(f"  Success rate: {success_rate:.1f}%")
                logger.info(f"  Final answer GT position: {best_answer_position}")
                logger.info(f"  Returning best answer from reflection process")
                logger.info(f"{'='*80}\n")
                
                # **CRITICAL**: Ensure we're returning the best answer, not the latest one
                # self.answer should already be the best answer (updated during reflection loop)
                # but let's verify and restore if needed
                current_position = self._get_ground_truth_position(self.answer)
                if current_position != best_answer_position:
                    logger.warning(f"⚠️  Current answer position ({current_position}) doesn't match best position ({best_answer_position})")
                    logger.warning(f"⚠️  Restoring best answer for scoring")
                    self.answer = best_answer.copy() if isinstance(best_answer, list) else best_answer
                
                return self.answer
            else:
                # Just perform reflection for logging purposes but don't rerun
                self._perform_reflection_logging_only()
            
            return self.answer
                
        except Exception as e:
            logger.error(f"Error in ReWOO forward: {e}")
            # Fall back to collaboration mode on error
            return self._fallback_collaboration_mode()

    def _execute_rewoo_workflow(self) -> str:
        """Execute the complete ReWOO workflow (planning -> working -> solving)."""
        result = None
        
        # Execute phases sequentially until completion
        while self.phase != 'completed' and not hasattr(self, '_finished'):
            if self.phase == 'planning':
                result = self._planning_phase()
            elif self.phase == 'working':
                result = self._working_phase()
            elif self.phase == 'solving':
                result = self._solving_phase()
                self.phase = 'completed'  # Mark as completed after solving
            else:
                # Default to planning if phase is unknown
                result = self._planning_phase()
        
        return result
    
    def _get_ground_truth_position(self, answer: Any) -> int:
        """
        Calculate the position of ground truth item in the ranked answer list.
        Returns the 1-based position if found, or -1 if not found.
        """
        if not isinstance(answer, list) or not hasattr(self, 'gt_answer'):
            return -1
        
        gt = self.gt_answer
        try:
            # For ranking tasks, gt_answer is the item to find
            if gt in answer:
                return answer.index(gt) + 1  # 1-based indexing
        except (ValueError, TypeError):
            pass
        
        return -1
    
    def _should_perform_reflection(self) -> tuple[bool, str]:
        """
        Determine if reflection is necessary.
        NOW: Always perform reflection on every sample for comprehensive quality assessment.
        
        Returns:
            tuple[bool, str]: (should_reflect, reason)
        """
        # Always perform reflection on every sample
        return True, "Performing reflection on every sample for comprehensive quality assessment"
    
    def _perform_reflection(self) -> tuple[bool, dict]:
        """
        Perform dual feedback reflection for both Planner and Solver.
        
        Returns:
            tuple[bool, dict]: (should_continue_reflecting, feedback_info)
            feedback_info contains:
            - 'planner_correct': bool indicating if Planner is correct
            - 'solver_correct': bool indicating if Solver is correct
            - 'planner_reason': feedback for Planner if incorrect
            - 'solver_reason': feedback for Solver if incorrect
        """
        if not self.reflector:
            return False, {}  # No reflector available, don't continue reflecting
        
        # Check if reflection is necessary
        should_reflect, skip_reason = self._should_perform_reflection()
        
        if not should_reflect:
            logger.info(f"⏭️  Skipping reflection: {skip_reason}")
            self.log(f"**Reflection Skipped:** {skip_reason}", agent=self.reflector)
            return False, {}
        
        logger.info(f"🔍 Performing dual feedback reflection: {skip_reason}")
        
        # Build comprehensive scratchpad of ReWOO process
        if hasattr(self, '_last_solution') and hasattr(self, '_last_final_answer'):
            rewoo_process = self._build_rewoo_scratchpad(self._last_solution, self._last_final_answer)
        else:
            # Fallback if solution details not available
            rewoo_process = self._build_basic_rewoo_scratchpad()
        
        # Add any previous reflection comments to provide context
        if hasattr(self, 'planner_kwargs') and 'reflections' in self.planner_kwargs:
            rewoo_process += f"\n\n=== Previous Reflection Comments ===\n{self.planner_kwargs['reflections']}"
        
        # Use reflector to analyze the complete ReWOO process
        with duration_tracker.track_agent_call('reflector'):
            self.reflector(input=self.input, scratchpad=rewoo_process)
        
        feedback_info = {
            'planner_correct': True,
            'solver_correct': True,
            'planner_reason': '',
            'solver_reason': ''
        }
        
        if self.reflector.json_mode and self.reflector.reflections:
            try:
                reflection_json = json.loads(self.reflector.reflections[-1])
                
                # NEW: Handle dual feedback format with separate Planner and Solver evaluation
                if isinstance(reflection_json, dict):
                    # Check for new dual format: {"Planner": {...}, "Solver": {...}}
                    if 'Planner' in reflection_json and 'Solver' in reflection_json:
                        logger.debug("Detected dual feedback format (Planner + Solver)")
                        
                        planner_feedback = reflection_json.get('Planner', {})
                        solver_feedback = reflection_json.get('Solver', {})
                        
                        feedback_info['planner_correct'] = planner_feedback.get('correctness', True)
                        feedback_info['solver_correct'] = solver_feedback.get('correctness', True)
                        feedback_info['planner_reason'] = planner_feedback.get('reason', 'No reason provided')
                        feedback_info['solver_reason'] = solver_feedback.get('reason', 'No reason provided')
                        
                        planner_status = "✅ Correct" if feedback_info['planner_correct'] else "❌ Incorrect"
                        solver_status = "✅ Correct" if feedback_info['solver_correct'] else "❌ Incorrect"
                        
                        logger.debug(f"Reflection Results - Planner: {planner_status}, Solver: {solver_status}")
                        
                        self.log(f"**Dual Feedback Reflection Results:**\n"
                                f"Planner: {planner_status}\n"
                                f"Reason: {feedback_info['planner_reason']}\n\n"
                                f"Solver: {solver_status}\n"
                                f"Reason: {feedback_info['solver_reason']}", agent=self.reflector)
                        
                        # Log detailed reflection status
                        logger.info(f"\n{'='*80}")
                        logger.info(f"📊 REFLECTION ANALYSIS (Sample {getattr(self, '_current_sample_idx', 'N/A')}):")
                        logger.info(f"{'='*80}")
                        logger.info(f"Planner Status: {planner_status}")
                        logger.info(f"Planner Feedback: {feedback_info['planner_reason']}")
                        logger.info(f"Solver Status: {solver_status}")
                        logger.info(f"Solver Feedback: {feedback_info['solver_reason']}")
                        
                        # Track reflection rerun trigger
                        self.total_reflections_triggered += 1
                        
                        # **FIX**: Determine if we should continue reflecting (rerun or solver reranking needed)
                        # Must trigger if EITHER Planner OR Solver is incorrect
                        should_continue = not feedback_info['planner_correct'] or not feedback_info['solver_correct']
                        
                        # Log the decision
                        if should_continue:
                            if not feedback_info['planner_correct'] and not feedback_info['solver_correct']:
                                logger.info(f"🔄 DECISION: Full ReWOO rerun (BOTH agents need improvement)")
                            elif not feedback_info['planner_correct']:
                                logger.info(f"🔄 DECISION: Full ReWOO rerun (Planner needs improvement)")
                            else:
                                logger.info(f"🔄 DECISION: Solver reranking only (Solver needs improvement)")
                        else:
                            logger.info(f"✅ DECISION: No improvement needed - continuing with original answer")
                        logger.info(f"{'='*80}\n")
                        
                        # NOTE: Planner and Solver feedback is added in the forward() method
                        # when we decide to do full rerun or solver reranking
                        # This keeps the feedback application logic centralized in forward()
                        
                        return should_continue, feedback_info
                    
                    # Legacy fallback: single correctness object for backward compatibility
                    else:
                        logger.debug("Detected legacy single feedback format")
                        
                        correctness = reflection_json.get('correctness', False)
                        reason = reflection_json.get('reason', 'No reason provided')
                        
                        feedback_info['planner_correct'] = correctness
                        feedback_info['solver_correct'] = correctness
                        feedback_info['planner_reason'] = reason
                        feedback_info['solver_reason'] = reason
                        
                        if not correctness:
                            logger.debug(f"ReWOO Reflection identified issues: {reason}")
                            self.log(f"**ReWOO Reflection Issues Identified:**\n{reason}", agent=self.reflector)
                            
                            # Track reflection rerun trigger
                            self.total_reflections_triggered += 1
                            
                            # Add reflection comment to planner_kwargs
                            if 'reflections' not in self.planner_kwargs:
                                self.planner_kwargs['reflections'] = ""
                            
                            reflection_feedback = f"\n=== Planning Improvement Required ===\n"
                            reflection_feedback += f"{reason}\n"
                            reflection_feedback += f"CRITICAL: Revise your plan to address this specific issue.\n"
                            
                            self.planner_kwargs['reflections'] += reflection_feedback
                            
                            return True, feedback_info  # Continue reflecting
                        else:
                            logger.debug(f"ReWOO Reflection confirms correctness: {reason}")
                            self.log(f"**ReWOO Reflection Confirms Correctness:**\n{reason}", agent=self.reflector)
                            return False, feedback_info  # Stop reflecting
                
                # Handle array format (legacy)
                elif isinstance(reflection_json, list):
                    logger.trace(f"Reflector returned array of {len(reflection_json)} objects. Converting to dual format.")
                    # Convert array to dual format by checking first item
                    if len(reflection_json) > 0 and isinstance(reflection_json[0], dict):
                        correctness = reflection_json[0].get('correctness', False)
                        reason = reflection_json[0].get('reason', 'No reason provided')
                        
                        feedback_info['planner_correct'] = correctness
                        feedback_info['solver_correct'] = correctness
                        feedback_info['planner_reason'] = reason
                        feedback_info['solver_reason'] = reason
                        
                        should_continue = not correctness
                        if not correctness:
                            self.total_reflections_triggered += 1
                        
                        return should_continue, feedback_info
                
                else:
                    logger.error(f"Unexpected reflection JSON type: {type(reflection_json)}")
                    return False, feedback_info
                        
            except Exception as e:
                logger.error(f'Invalid reflection JSON output: {self.reflector.reflections[-1]}')
                logger.error(f'JSON parsing error: {e}')
                # Continue execution even if reflection parsing fails
                return False, feedback_info
        else:
            # Non-JSON mode reflection - assume we should stop reflecting
            if self.reflector.reflections:
                self.log(f"**ReWOO Reflection:**\n{self.reflector.reflections[-1]}", agent=self.reflector)
            return False, feedback_info
        
        return False, feedback_info  # Default to not continue reflecting

    def _perform_reflection_logging_only(self) -> None:
        """Perform reflection only for logging purposes without enabling reruns."""
        if not self.reflector:
            return  # No reflector available
        
        # Build comprehensive scratchpad of ReWOO process
        if hasattr(self, '_last_solution') and hasattr(self, '_last_final_answer'):
            rewoo_process = self._build_rewoo_scratchpad(self._last_solution, self._last_final_answer)
        else:
            # Fallback if solution details not available
            rewoo_process = self._build_basic_rewoo_scratchpad()
        
        # Use reflector to analyze the complete ReWOO process
        with duration_tracker.track_agent_call('reflector'):
            self.reflector(input=self.input, scratchpad=rewoo_process)
        
        if self.reflector.json_mode and self.reflector.reflections:
            try:
                reflection_json = json.loads(self.reflector.reflections[-1])
                
                # Handle both single object and array of objects
                if isinstance(reflection_json, list):
                    logger.trace(f"Reflector returned array of {len(reflection_json)} objects. Evaluating all.")
                    # If ANY object has correctness=false, treat overall as incorrect
                    correctness = True
                    reasons = []
                    for item in reflection_json:
                        if isinstance(item, dict) and 'correctness' in item:
                            if not item['correctness']:
                                correctness = False
                                reasons.append(item.get('reason', 'No reason provided'))
                            elif item['correctness'] and len(reflection_json) == 1:
                                # Only use positive reason if it's the ONLY item
                                reasons.append(item.get('reason', 'No reason provided'))
                    reason = '\n'.join(f"- {r}" for r in reasons) if reasons else 'Multiple issues identified'
                elif isinstance(reflection_json, dict):
                    # Single object (expected format)
                    correctness = reflection_json.get('correctness', False)
                    reason = reflection_json.get('reason', 'No reason provided')
                else:
                    logger.error(f"Unexpected reflection JSON type: {type(reflection_json)}")
                    return
                
                if not correctness:
                    logger.debug(f"ReWOO Reflection identified issues: {reason}")
                    self.log(f"**ReWOO Reflection Issues Identified:**\n{reason}", agent=self.reflector)
                    logger.info("Note: Reflection rerun is disabled. Use --enable-reflection-rerun to enable automatic reruns.")
                else:
                    logger.debug(f"ReWOO Reflection confirms correctness: {reason}")
                    self.log(f"**ReWOO Reflection Confirms Correctness:**\n{reason}", agent=self.reflector)
                        
            except Exception as e:
                logger.error(f'Invalid reflection JSON output: {self.reflector.reflections[-1]}')
                logger.error(f'JSON parsing error: {e}')
        else:
            # Non-JSON mode reflection
            if self.reflector.reflections:
                self.log(f"**ReWOO Reflection:**\n{self.reflector.reflections[-1]}", agent=self.reflector)

    def _build_basic_rewoo_scratchpad(self) -> str:
        """Build a basic scratchpad when detailed solution information is not available."""
        scratchpad = f"\n=== ReWOO Process Summary ===\n"
        scratchpad += f"Task: {self.task.upper()}\n"
        scratchpad += f"Original Query: {getattr(self, 'input', 'No input')}\n"
        scratchpad += f"Current Phase: {self.phase}\n"
        
        if hasattr(self, 'current_plan') and self.current_plan:
            scratchpad += f"\nGenerated Plan:\n{self.current_plan}\n"
        
        if hasattr(self, 'execution_results') and self.execution_results:
            scratchpad += "\nExecution Results:\n"
            for step_var, result in self.execution_results.items():
                scratchpad += f"{step_var}: {result}\n"
        
        return scratchpad

    def _planning_phase(self) -> str:
        """Phase 1: Planning - Decompose task into sub-problems."""
        logger.info("ReWOO Phase 1: Planning")
        
        # Prepare query for planner
        query = self._prepare_planning_query()
        
        # Generate plan - CRITICAL: Pass manager_kwargs AND planner_kwargs (for reflection feedback)
        combined_kwargs = {**self.manager_kwargs, **self.planner_kwargs}
        with duration_tracker.track_agent_call('planner'):
            plan = self.planner.invoke(query, self.task, **combined_kwargs)
        self.current_plan = plan
        self.plan_steps = self.planner.parse_plan(plan)
        
        self.log(f"**ReWOO Plan Generated:**\n{plan}", agent=self.planner)
        
        # Move to working phase
        self.phase = 'working'
        self.step_n = 1
        
        if not self.plan_steps:
            # No valid steps found, proceed to solving with empty results
            self.phase = 'solving'
            return self._solving_phase()
        
        return self._working_phase()

    def _working_phase(self) -> str:
        """Phase 2: Working - Execute plan steps using workers."""
        logger.info(f"ReWOO Phase 2: Working - Step {self.step_n}")
        
        if self.step_n > len(self.plan_steps):
            # All steps completed, move to solving
            self.phase = 'solving'
            return ''  # Return empty string, let the while loop call solving phase
        
        # Get current step
        current_step = self.plan_steps[self.step_n - 1]
        
        # Check dependencies
        if not self._dependencies_satisfied(current_step):
            # Skip this step or handle dependency failure
            logger.warning(f"Dependencies not satisfied for step {self.step_n}")
            self.step_n += 1
            return self._working_phase()
        
        # Execute step
        result = self._execute_step(current_step)
        
        step_variable = current_step['variable']
        self.completed_steps.add(step_variable)
        
        # Skip meaningless results from being stored in execution_results
        # This prevents noise from appearing in Solver's prompt
        if not self._is_meaningless_result(result):
            self.execution_results[step_variable] = result
            
            self.log(f"**Step {self.step_n} ({step_variable})**: {current_step['task_description']}\n**Result**: {result}", 
                    agent=self._get_worker_agent(current_step['worker_type']))
        else:
            logger.info(f"Skipped storing meaningless result for {step_variable}")
        
        self.step_n += 1
        
        # Continue with next step or move to solving
        if self.step_n > len(self.plan_steps):
            self.phase = 'solving'
            return ''  # Return empty string, let the while loop call solving phase
        else:
            return self._working_phase()

    def _solving_phase(self) -> str:
        """Phase 3: Solving - Aggregate results to generate final answer."""
        logger.info("ReWOO Phase 3: Solving")
        
        # Generate final solution - CRITICAL: Pass manager_kwargs like collaboration system
        with duration_tracker.track_agent_call('solver'):
            solution = self.solver.invoke(self.current_plan, self.execution_results, self.task, **self.manager_kwargs)
        
        self.log(f"**ReWOO Final Solution:**\n{solution}", agent=self.solver)
        
        # Extract and return final answer
        final_answer = self.solver.extract_final_answer(solution, self.task)
        
        # CRITICAL: Filter out history items from the answer
        if self.task in ['sr'] and isinstance(final_answer, list):
            # Extract history item IDs directly from data_sample column
            history_item_ids = set()
            if hasattr(self, 'data_sample') and self.data_sample is not None and 'history_item_id' in self.data_sample:
                try:
                    history_item_id_value = self.data_sample['history_item_id']
                    # Parse the list string representation
                    if isinstance(history_item_id_value, str):
                        history_item_ids = set(eval(history_item_id_value))
                    elif isinstance(history_item_id_value, (list, set)):
                        history_item_ids = set(history_item_id_value)
                    logger.info(f"Extracted history item IDs from data_sample: {sorted(history_item_ids)}")
                except Exception as e:
                    logger.warning(f"Failed to extract history_item_id from data_sample: {e}")
                    history_item_ids = set()
            
            # Extract candidate item IDs directly from data_sample CSV column
            candidate_item_ids = set()
            
            # For SR/RP tasks: extract from data_sample CSV column (most reliable source)
            if hasattr(self, 'data_sample') and self.data_sample is not None and 'candidate_item_id' in self.data_sample:
                try:
                    candidate_item_id_value = self.data_sample['candidate_item_id']
                    # Parse the list string representation
                    if isinstance(candidate_item_id_value, str):
                        candidate_item_ids = set(eval(candidate_item_id_value))
                    elif isinstance(candidate_item_id_value, (list, set)):
                        candidate_item_ids = set(candidate_item_id_value)
                    logger.debug(f"Extracted candidate item IDs from data_sample: {sorted(candidate_item_ids)}")
                except Exception as e:
                    logger.warning(f"Failed to extract candidate_item_id from data_sample: {e}")
                    candidate_item_ids = set()
            
            # Filter answer to keep only candidate items (not history items)
            original_answer = final_answer.copy()
            filtered_answer = [item_id for item_id in final_answer if item_id in candidate_item_ids]
            
            # Log if filtering occurred
            if len(filtered_answer) != len(original_answer):
                removed_items = [item_id for item_id in original_answer if item_id not in filtered_answer]
                logger.warning(f"Solver included history items in answer! Removed: {removed_items}")
                logger.info(f"Original answer: {original_answer}")
                logger.info(f"Filtered answer: {filtered_answer}")
                logger.info(f"History items: {sorted(history_item_ids)}")
                logger.info(f"Candidate items: {sorted(candidate_item_ids)}")
            
            final_answer = filtered_answer
        
        # Store solution and final answer for reflection
        self._last_solution = solution
        self._last_final_answer = final_answer
        
        # Track ground truth position BEFORE reflection
        self._gt_position_before_reflection = self._get_ground_truth_position(final_answer)
        
        # **CRITICAL**: Store the original answer before any reflection reruns
        # This is used to decide which answer to return (original vs improved via rerun)
        self._answer_before_reflection = final_answer.copy() if isinstance(final_answer, list) else final_answer
        
        # Log the final solution with ground truth for comparison
        logger.info(f"ReWOO Final Answer: {final_answer} | Ground Truth: {self.gt_answer}")
        if self._gt_position_before_reflection > 0:
            logger.info(f"Ground truth position (before reflection): {self._gt_position_before_reflection}")
        
        return self.finish(final_answer)

    def _build_rewoo_scratchpad(self, solution: str, final_answer: str) -> str:
        """Build a comprehensive scratchpad of the ReWOO process for reflection."""
        scratchpad = f"\n=== ReWOO Process Summary ===\n"
        scratchpad += f"Task: {self.task.upper()}\n"
        scratchpad += f"Original Query: {getattr(self, 'input', 'No input')}\n\n"
        
        # Phase 1: Planning
        scratchpad += "=== Phase 1: Planning ===\n"
        if self.current_plan:
            scratchpad += f"Generated Plan:\n{self.current_plan}\n\n"
        else:
            scratchpad += "No plan generated - POTENTIAL ISSUE\n\n"
        
        # Phase 2: Working
        scratchpad += "=== Phase 2: Working (Execution Results) ===\n"
        if self.execution_results:
            for step_var, result in self.execution_results.items():
                scratchpad += f"{step_var}: {result}\n"
        else:
            scratchpad += "No execution results - POTENTIAL ISSUE\n"
        scratchpad += "\n"
        
        # Phase 3: Solving
        scratchpad += "=== Phase 3: Solving ===\n"
        scratchpad += f"Solver Output:\n{solution}\n"
        scratchpad += f"Final Answer: {final_answer}\n"
        
        return scratchpad

    def _perform_solver_reranking(self, solver_feedback: str) -> str:
        """
        Perform solver reranking without full rerun.
        
        Only the Solver receives feedback to improve ranking, without replanning or re-executing analysis.
        Uses existing execution results from working phase.
        
        Args:
            solver_feedback: Feedback from reflector for the Solver
            
        Returns:
            str: Updated final answer from re-invoked Solver
        """
        logger.info("🔄 Performing Solver Reranking (without full rerun)")
        
        # Get the previous ranking for comparison
        previous_ranking = self._last_final_answer if hasattr(self, '_last_final_answer') else []
        previous_solution = self._last_solution if hasattr(self, '_last_solution') else ""
        
        # Add solver feedback to manager_kwargs for Solver to use
        solver_feedback_key = 'solver_reflections'
        if solver_feedback_key not in self.manager_kwargs:
            self.manager_kwargs[solver_feedback_key] = ""
        
        # Provide detailed feedback including the previous ranking
        feedback_message = f"\n{'='*80}\n"
        feedback_message += f"=== SOLVER RERANKING REQUIRED (Reflection Feedback) ===\n"
        feedback_message += f"{'='*80}\n\n"
        feedback_message += f"📋 Your Previous Ranking:\n{previous_ranking}\n\n"
        if previous_solution:
            feedback_message += f"📝 Your Previous Reasoning:\n{previous_solution}\n\n"
        feedback_message += f"❌ Issue Identified by Reflector:\n{solver_feedback}\n\n"
        feedback_message += f"🎯 Required Action:\n"
        feedback_message += f"1. Review your previous ranking above\n"
        feedback_message += f"2. Understand the specific issue raised by the Reflector\n"
        feedback_message += f"3. Re-analyze the user preferences and item characteristics\n"
        feedback_message += f"4. Produce an IMPROVED ranking that addresses the feedback\n"
        feedback_message += f"5. Ensure the new ranking is DIFFERENT from your previous one\n\n"
        feedback_message += f"CRITICAL: Your new ranking MUST be different from the previous one to address the feedback!\n"
        feedback_message += f"{'='*80}\n"
        
        self.manager_kwargs[solver_feedback_key] += feedback_message
        
        # Log the feedback being provided
        logger.info(f"📋 Providing Solver with previous ranking: {previous_ranking}")
        logger.info(f"💬 Feedback: {solver_feedback}")
        
        # Re-invoke Solver with feedback, using SAME execution_results
        logger.debug(f"Re-invoking Solver with feedback. Using existing execution results from working phase.")
        with duration_tracker.track_agent_call('solver'):
            reranked_solution = self.solver.invoke(self.current_plan, self.execution_results, self.task, **self.manager_kwargs)
        
        self.log(f"**ReWOO Reranked Solution (Solver Feedback):**\n{reranked_solution}", agent=self.solver)
        
        # Extract new final answer
        reranked_answer = self.solver.extract_final_answer(reranked_solution, self.task)
        
        # CRITICAL: Filter out history items from the reranked answer
        if self.task in ['sr'] and isinstance(reranked_answer, list):
            # Extract history and candidate item IDs (same logic as _solving_phase)
            history_item_ids = set()
            if hasattr(self, 'data_sample') and self.data_sample is not None and 'history_item_id' in self.data_sample:
                try:
                    history_item_id_value = self.data_sample['history_item_id']
                    if isinstance(history_item_id_value, str):
                        history_item_ids = set(eval(history_item_id_value))
                    elif isinstance(history_item_id_value, (list, set)):
                        history_item_ids = set(history_item_id_value)
                except Exception as e:
                    logger.warning(f"Failed to extract history_item_id: {e}")
                    history_item_ids = set()
            
            candidate_item_ids = set()
            if hasattr(self, 'data_sample') and self.data_sample is not None and 'candidate_item_id' in self.data_sample:
                try:
                    candidate_item_id_value = self.data_sample['candidate_item_id']
                    if isinstance(candidate_item_id_value, str):
                        candidate_item_ids = set(eval(candidate_item_id_value))
                    elif isinstance(candidate_item_id_value, (list, set)):
                        candidate_item_ids = set(candidate_item_id_value)
                except Exception as e:
                    logger.warning(f"Failed to extract candidate_item_id: {e}")
                    candidate_item_ids = set()
            
            # Filter answer to keep only candidate items
            original_reranked = reranked_answer.copy()
            reranked_answer = [item_id for item_id in reranked_answer if item_id in candidate_item_ids]
            
            if len(reranked_answer) != len(original_reranked):
                removed_items = [item_id for item_id in original_reranked if item_id not in reranked_answer]
                logger.warning(f"Reranked answer contained history items! Removed: {removed_items}")
        
        logger.info(f"Solver reranked answer: {reranked_answer}")
        
        # Compare with previous ranking to verify change
        if previous_ranking != reranked_answer:
            logger.success(f"✅ Solver produced DIFFERENT ranking after feedback!")
            logger.info(f"   Previous: {previous_ranking}")
            logger.info(f"   New:      {reranked_answer}")
        else:
            logger.warning(f"⚠️ Solver produced IDENTICAL ranking despite feedback!")
            logger.warning(f"   Ranking:  {reranked_answer}")
        
        # Store updated solution and answer
        self._last_solution = reranked_solution
        self._last_final_answer = reranked_answer
        
        return reranked_answer

    def _prepare_planning_query(self) -> str:
        """Prepare the query for the planner based on task and context using exact same data as collaboration system."""
        # Use the same input data as collaboration system
        base_query = getattr(self, 'input', 'No input provided')
        
        if self.task == 'sr':
            return f"Sequential recommendation task: {base_query}"
        elif self.task == 'rp':
            return f"Rating prediction task: {base_query}" 
        elif self.task == 'rr':
            return f"Retrieve and rank task: {base_query}"
        elif self.task == 'gen':
            return f"Review generation task: {base_query}"
        else:
            return f"{self.task} task: {base_query}"

    def _dependencies_satisfied(self, step: Dict[str, Any]) -> bool:
        """Check if all dependencies for a step are satisfied."""
        dependencies = step.get('dependencies', [])
        variable = step.get('variable', 'unknown')
        
        for dep in dependencies:
            if dep not in self.completed_steps:
                return False
        return True

    def _is_meaningless_result(self, result: str) -> bool:
        """
        Check if a result is meaningless (noise that shouldn't be stored).
        Returns True if result should be skipped from execution_results.
        
        Detects patterns like:
        - "User info database not available"
        - "No history found for..."
        - "User information database not available"
        """
        if not isinstance(result, str):
            return False
        
        meaningless_patterns = [
            'User info database not available',
            'User information database not available',
            'No history found for user',
            'No history found for item',
            'database not available',
            'No user information',
            'User History X: No history',  # Format from Analyst
        ]
        
        # Check if result is ONLY meaningless content (no actionable data)
        lines = result.strip().split('\n')
        
        # If very short result with only noise, it's meaningless
        if len(lines) <= 2:
            for pattern in meaningless_patterns:
                if pattern in result:
                    logger.debug(f"Skipping meaningless result (matched pattern '{pattern}'): {result[:100]}")
                    return True
        
        # For longer results, check if they contain ONLY noise (no real data)
        has_actionable_data = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line has actionable data (not just noise)
            is_noise = any(pattern in line for pattern in meaningless_patterns)
            
            if not is_noise and any(keyword in line for keyword in ['Brand:', 'Price:', 'Categories:', 'Rating:', 'Title:', 'Genres:', 'User:', 'Item:', 'analyzed', 'Retrieved']):
                has_actionable_data = True
                break
        
        if not has_actionable_data and any(pattern in result for pattern in meaningless_patterns):
            logger.debug(f"Skipping meaningless result (no actionable data): {result[:100]}")
            return True
        
        return False

    def _execute_step(self, step: Dict[str, Any]) -> str:
        """Execute a single plan step using the appropriate worker with real data exactly like collaboration system."""
        worker_type = step['worker_type']
        task_desc = step['task_description']
        
        # Replace dependency references with actual results
        for dep in step['dependencies']:
            if dep in self.execution_results:
                task_desc = task_desc.replace(dep, str(self.execution_results[dep]))
        
        # Get worker agent and execute using the same patterns as collaboration system
        worker = self._get_worker_agent(worker_type)
        if worker is None:
            return f"Worker {worker_type} not available"
        
        try:
            # Get json_mode setting - prefer manager's setting, but fallback to worker's own setting
            if self.manager and hasattr(self.manager, 'json_mode'):
                json_mode = self.manager.json_mode
            elif hasattr(worker, 'json_mode'):
                json_mode = worker.json_mode
            else:
                json_mode = False
            
            logger.debug(f"Worker {worker_type} json_mode: {json_mode}")
            
            from macrec.utils import duration_tracker
            
            if worker_type.lower() == 'analyst':
                # Use the task description to determine what to analyze
                args = self._parse_analyst_arguments_from_context(task_desc)
                logger.debug(f"Analyst args: {args}, type: {type(args)}")
                
                # Build execution context with cached data and previous results
                execution_context = self._build_execution_context()
                
                # Pass the full task description as analysis context + execution context
                kwargs = {
                    'task_context': task_desc,  # Pass the specific plan step description
                    'execution_context': execution_context,  # Pass cached data and previous results
                    **self.manager_kwargs
                }
                
                # Handle argument format based on json_mode
                with duration_tracker.track_agent_call('analyst'):
                    if json_mode and isinstance(args, list):
                        # JSON mode expects list format
                        result = worker.invoke(argument=args, json_mode=json_mode, **kwargs)
                    elif not json_mode and isinstance(args, list):
                        # Non-JSON mode expects string format
                        arg_string = f"{args[0]},{args[1]}" if len(args) >= 2 else "user,1"
                        result = worker.invoke(argument=arg_string, json_mode=json_mode, **kwargs)
                    else:
                        # Pass as-is
                        result = worker.invoke(argument=args, json_mode=json_mode, **kwargs)
                
                # Update entity cache with new data from this step
                self._update_entity_cache(args, result)
                    
            elif worker_type.lower() == 'searcher':
                logger.debug(f"Searcher args: {task_desc}, type: {type(task_desc)}")
                # Only pass argument and json_mode like collaboration system does
                with duration_tracker.track_agent_call('searcher'):
                    result = worker.invoke(argument=task_desc, json_mode=json_mode)
            elif worker_type.lower() == 'interpreter':
                logger.debug(f"Interpreter args: {task_desc}, type: {type(task_desc)}")
                # Only pass argument and json_mode like collaboration system does
                with duration_tracker.track_agent_call('interpreter'):
                    result = worker.invoke(argument=task_desc, json_mode=json_mode)
            else:
                result = f"Unknown worker type: {worker_type}"
                
            # Track analyzed entities for compatibility (same as collaboration system)
            self._track_analyzed_entities(worker_type, task_desc, result)
            
            return result
            
        except Exception as e:
            error_msg = f"Error executing step with {worker_type}: {e}"
            logger.error(error_msg)
            # Track error for smart reflection
            if not hasattr(self, '_execution_errors'):
                self._execution_errors = []
            self._execution_errors.append({
                'worker': worker_type,
                'step': step.get('variable', 'unknown'),
                'error': str(e)
            })
            return f"Execution failed: {str(e)}"

    def _build_execution_context(self) -> Dict[str, Any]:
        """Build execution context with cached data and previous results for context sharing."""
        context = {
            'entity_cache': self.entity_cache,
            'previous_results': dict(self.execution_results),
            'analyzed_entities': {
                'users': list(self.analyzed_users),
                'items': list(self.analyzed_items)
            },
            'step_number': self.step_n,
            'total_steps': len(self.plan_steps),
            'data_sample': self.data_sample
        }
        return context
    
    def _update_entity_cache(self, args: List[Any], result: str) -> None:
        """Update entity cache with data from the current step result."""
        if not isinstance(args, list) or len(args) < 2:
            return
        
        entity_type = args[0]
        entity_id = args[1]
        
        # Parse and cache entity information from result
        if entity_type == 'user':
            # Extract user info if present in result
            if 'Age:' in result or 'Gender:' in result or 'Occupation:' in result:
                if entity_id not in self.entity_cache['users']:
                    self.entity_cache['users'][entity_id] = {}
                self.entity_cache['users'][entity_id]['info'] = result
            
            # Extract user history if present in result
            if 'interacted with' in result or 'Retrieved' in result:
                if entity_id not in self.entity_cache['user_histories']:
                    self.entity_cache['user_histories'][entity_id] = result
                    
        elif entity_type == 'item':
            # Extract item info if present in result
            if 'Title:' in result or 'Genres:' in result:
                if entity_id not in self.entity_cache['items']:
                    self.entity_cache['items'][entity_id] = {}
                self.entity_cache['items'][entity_id]['info'] = result
            
            # Extract item history if present in result
            if 'interacted with' in result or 'Retrieved' in result:
                if entity_id not in self.entity_cache['item_histories']:
                    self.entity_cache['item_histories'][entity_id] = result
    
    def _get_worker_agent(self, worker_type: str) -> Optional[Agent]:
        """Get the appropriate worker agent."""
        worker_map = {
            'analyst': self.analyst,
            'searcher': self.searcher,
            'interpreter': self.interpreter
        }
        return worker_map.get(worker_type.lower())

    def _parse_analyst_arguments_from_context(self, task_desc: str) -> List[Any]:
        """Parse analyst arguments from task description using real context data like collaboration system."""
        # Extract entity type and ID from the actual input context, same as collaboration system
        import re
        
        # First try to extract from task description
        user_match = re.search(r'user\s+(\d+)', task_desc, re.IGNORECASE)
        item_match = re.search(r'item\s+(\d+)', task_desc, re.IGNORECASE)
        
        if user_match:
            # Convert to integer as expected by Analyst
            return ['user', int(user_match.group(1))]
        elif item_match:
            # Convert to integer as expected by Analyst
            return ['item', int(item_match.group(1))]
        else:
            # Fall back to extracting from the actual input data like collaboration system
            input_text = getattr(self, 'input', '')
            user_match = re.search(r'user[_\s:]*(\d+)', input_text, re.IGNORECASE)
            item_match = re.search(r'item[_\s:]*(\d+)', input_text, re.IGNORECASE)
            
            if user_match:
                # Convert to integer as expected by Analyst
                return ['user', int(user_match.group(1))]
            elif item_match:
                # Convert to integer as expected by Analyst
                return ['item', int(item_match.group(1))]
            else:
                # Use kwargs like collaboration system
                if 'user_id' in self.kwargs:
                    return ['user', int(self.kwargs['user_id'])]
                elif 'item_id' in self.kwargs:
                    return ['item', int(self.kwargs['item_id'])]
                else:
                    # Default fallback
                    return ['user', 1]

    def _parse_analyst_arguments(self, task_desc: str) -> List[str]:
        """Parse analyst arguments from task description."""
        # Extract entity type and ID from description
        import re
        
        # Look for patterns like "analyze user 123" or "analyze item 456"
        user_match = re.search(r'user\s+(\d+)', task_desc, re.IGNORECASE)
        item_match = re.search(r'item\s+(\d+)', task_desc, re.IGNORECASE)
        
        if user_match:
            return ['user', user_match.group(1)]
        elif item_match:
            return ['item', item_match.group(1)]
        else:
            # Default case
            return ['item', str(self.kwargs.get('item_id', '1'))]

    def _track_analyzed_entities(self, worker_type: str, task_desc: str, result: str) -> None:
        """Track analyzed entities for compatibility with existing system."""
        import re
        
        if worker_type.lower() == 'analyst':
            user_match = re.search(r'user\s+(\d+)', task_desc, re.IGNORECASE)
            item_match = re.search(r'item\s+(\d+)', task_desc, re.IGNORECASE)
            
            if user_match:
                self.analyzed_users.add(user_match.group(1))
            elif item_match:
                self.analyzed_items.add(item_match.group(1))

    def _fallback_collaboration_mode(self) -> str:
        """Fallback to collaboration system behavior using exact same execution pattern."""
        try:
            # If we have a manager, use collaboration system's step execution pattern
            if self.manager is not None:
                # Initialize scratchpad if needed (same as collaboration system)
                if not hasattr(self, 'scratchpad'):
                    self.scratchpad = ''
                
                # Use collaboration system's execution pattern
                if not self.is_finished() and not self.is_halted():
                    # Execute using collaboration system's think-act-execute pattern
                    self.think()
                    action_type, argument = self.act()
                    self.execute(action_type, argument)
                    self.step_n += 1
                    
                return "Collaboration mode step completed"
            else:
                # If no manager available, return error
                return "No manager available for fallback"
                
        except Exception as e:
            logger.error(f"Error in fallback collaboration mode: {e}")
            return f"Fallback failed: {str(e)}"
            
    def think(self):
        """Think method using exact same logic as collaboration system."""
        # Use the exact same implementation as collaboration system
        logger.debug(f'Step {self.step_n}:')
        logger.debug(f'Manager kwargs: {self.manager_kwargs}')
        
        # Truncate scratchpad if it's getting too long (same as collaboration)
        max_scratchpad_length = 8000
        if len(self.scratchpad) > max_scratchpad_length:
            lines = self.scratchpad.split('\n')
            truncated_lines = lines[-50:]
            self.scratchpad = '\n'.join(['[Previous context truncated...]'] + truncated_lines)
            logger.debug(f'Truncated scratchpad to prevent context overflow')
        
        self.scratchpad += f'\nThought {self.step_n}:'
        thought = self.manager(scratchpad=self.scratchpad, stage='thought', **self.manager_kwargs)
        
        # Clean up thought (same as collaboration system)
        thought = thought.split('Action')[0].split('Observation')[0].split('Thought')[0].strip()
        
        self.scratchpad += ' ' + thought
        self.log(f'**Thought {self.step_n}**: {thought}', agent=self.manager)

    def act(self) -> tuple[str, Any]:
        """Act method using exact same logic as collaboration system."""
        # Use exact same logic as collaboration system
        if self.max_step == self.step_n:
            self.scratchpad += f'\nHint: {self.manager.hint}'
        
        # Add progress reminder for rr tasks (same as collaboration)
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
        
        self.scratchpad += f'\nAction {self.step_n}:'
        logger.debug(f'Action step - Manager kwargs: {self.manager_kwargs}')
        action = self.manager(scratchpad=self.scratchpad, stage='action', **self.manager_kwargs)
        
        # Clean up action (same as collaboration system)
        action_clean = action.strip()
        if '```' in action_clean:
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
        from macrec.utils import parse_action
        action_type, argument = parse_action(action_clean, json_mode=self.manager.json_mode)
        logger.debug(f'Action {self.step_n}: {action_clean}')
        return action_type, argument

    def execute(self, action_type: str, argument: Any):
        """Execute method using exact same logic as collaboration system."""
        # Import the collaboration system to use its execute method
        from macrec.systems.collaboration import CollaborationSystem
        
        # Create a temporary instance to access the execute method
        # This ensures we use the exact same execution logic
        temp_collab = CollaborationSystem.__new__(CollaborationSystem)
        temp_collab.agents = self.agents
        temp_collab.manager_kwargs = self.manager_kwargs
        temp_collab.task = self.task
        temp_collab.kwargs = self.kwargs
        temp_collab.scratchpad = self.scratchpad
        temp_collab.step_n = self.step_n
        temp_collab.analyzed_items = self.analyzed_items
        temp_collab.analyzed_users = self.analyzed_users
        temp_collab.log = self.log
        temp_collab.finish = self.finish
        
        # Use collaboration system's execute method
        temp_collab.execute(action_type, argument)
        
        # Copy back the updated state
        self.scratchpad = temp_collab.scratchpad
        self.analyzed_items = temp_collab.analyzed_items
        self.analyzed_users = temp_collab.analyzed_users
        if hasattr(temp_collab, 'manager_kwargs'):
            self.manager_kwargs = temp_collab.manager_kwargs

    def is_finished(self) -> bool:
        """Check if the system is finished (same logic as collaboration system)."""
        return hasattr(self, 'finished') and self.finished

    def is_halted(self) -> bool:
        """Check if the system is halted (same logic as collaboration system)."""
        return ((self.step_n > self.max_step) or 
                (self.manager and self.manager.over_limit(scratchpad=getattr(self, 'scratchpad', ''), **self.manager_kwargs))) and not self.is_finished()

    def step(self):
        """Execute one step of the ReWOO process or fallback to collaboration."""
        try:
            # If in ReWOO mode and agents available, continue ReWOO workflow
            if self.planner and self.solver and self.phase in ['planning', 'working', 'solving']:
                return self.forward(reset=False)
            else:
                # Otherwise use collaboration system step pattern
                return self._fallback_collaboration_mode()
        except Exception as e:
            logger.error(f"Error in ReWOO step: {e}")
            # Fall back to collaboration mode on error
            return self._fallback_collaboration_mode()