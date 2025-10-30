import os
import jsonlines
from tqdm import tqdm
from typing import Any
from loguru import logger
from argparse import ArgumentParser

from macrec.tasks.generation import GenerationTask
from macrec.utils import str2list, NumpyEncoder, token_tracker
from macrec.evaluation import MetricDict, HitRatioAt, NDCGAt, RMSE, Accuracy, MAE

class EvaluateTask(GenerationTask):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser = GenerationTask.parse_task_args(parser)
        parser.add_argument('--steps', type=int, default=1, help='Number of steps')
        parser.add_argument('--topks', type=str2list, default=[1, 3, 5], help='Top-Ks for ranking task')
        return parser

    def get_metrics(self, topks: list[int] = [1, 3, 5]):
        if self.task == 'rp':
            self.metrics = MetricDict({
                'true_rmse': RMSE(),
                'true_mae': MAE(),
                'true_accuracy': Accuracy(),
                'valid_rmse': RMSE(),
                'valid_mae': MAE(),
            })
        elif self.task == 'sr' or self.task == 'rr':
            self.metrics = MetricDict({
                'true_hit_rate': HitRatioAt(topks=topks),
                'true_ndcg': NDCGAt(topks=topks),
                'valid_hit_rate': HitRatioAt(topks=topks),
                'valid_ndcg': NDCGAt(topks=topks),
            })
        else:
            raise NotImplementedError

    def update_evaluation(self, answer: float | int | str, gt_answer: float | int | str) -> str:
        valid = self.system.finished
        self.total_count += 1
        if valid:
            self.valid_count += 1
        logger.debug(f'Answer: {answer}, Ground Truth: {gt_answer}')
        if valid:
            return self.metrics.update(output={
                'answer': answer,
                'label': gt_answer,
            })
        else:
            return self.metrics.update(output={
                'answer': answer,
                'label': gt_answer,
            }, prefix='true')

    def _log_cumulative_scores(self, sample_id: str) -> None:
        """Log cumulative evaluation scores and token usage to the log file after each sample completes.
        
        Args:
            sample_id: The ID of the sample that just completed
        """
        try:
            if hasattr(self, 'log_handler_id') and self.log_handler_id is not None:
                # Find the log file path from logger handlers
                log_file_path = None
                for handler_id, handler in logger._core.handlers.items():
                    if handler_id == self.log_handler_id:
                        if hasattr(handler._sink, 'name'):
                            log_file_path = handler._sink.name
                        elif hasattr(handler._sink, '_file') and hasattr(handler._sink._file, 'name'):
                            log_file_path = handler._sink._file.name
                        break
                
                if log_file_path:
                    # Compute all metrics
                    result = self.metrics.compute()
                    
                    # Get current token stats
                    current_task_stats = token_tracker.get_task_stats()
                    total_input_tokens = current_task_stats.get('total_input_tokens', 0)
                    total_output_tokens = current_task_stats.get('total_output_tokens', 0)
                    total_tokens = current_task_stats.get('total_tokens', 0)
                    
                    # Write scores to the log file
                    with open(log_file_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"\n===== Sample {sample_id} - Cumulative Scores ({self.total_count} samples) =====\n")
                        log_file.write(f"Tokens: Input={total_input_tokens} | Output={total_output_tokens} | Total={total_tokens}\n\n")
                        
                        # Write each metric in the same format as metrics.report()
                        for metric_name, metric_values in result.items():
                            if len(metric_values) == 1:
                                # Single value metric
                                value = next(iter(metric_values.values()))
                                log_file.write(f"{metric_name}: {value:.4f}\n")
                            else:
                                # Multi-value metric
                                log_file.write(f"{metric_name}:\n")
                                for key, value in metric_values.items():
                                    log_file.write(f"  {key}: {value:.4f}\n")
                        log_file.write("\n")
        except Exception as e:
            logger.warning(f"Failed to log cumulative scores: {e}")

    @property
    def running_steps(self):
        return self.steps

    def before_generate(self) -> None:
        self.get_metrics(self.topks)
        # Initialize counters for tracking valid answers
        self.valid_count = 0
        self.total_count = 0
        self.failed_samples = []  # Track which samples failed
        self.gt_positions = []  # Track ground truth positions in ranked lists
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        dataset = os.path.basename(os.path.dirname(self.args.data_file))
        data_file = os.path.basename(self.args.data_file)
        run_dir = os.path.join(root_dir, 'run', dataset, self.task, self.args.system)
        os.makedirs(run_dir, exist_ok=True)
        output_args = {
            'data_file': data_file,
            'sampled': self.sampled if hasattr(self, 'sampled') else False,
            'config': self.args.system_config.replace('/', '-'),
            'max_his': self.args.max_his,
            'topk': self.topks,
        }
        output_file_name = '_'.join([f'{k}={v}' for k, v in output_args.items()]) + '.jsonl'
        self.output_file = jsonlines.open(os.path.join(run_dir, output_file_name), mode="w", dumps=NumpyEncoder(ensure_ascii=False).encode, flush=True)
        

    def after_step(self, answer: Any, gt_answer: int | float | str, step: int, record: dict) -> None:
        record[f'Answer_{step}'] = answer
        if hasattr(self.system, 'reflected') and self.system.reflected and self.system.reflector.keep_reflections:
            # assert isinstance(self.system, ReflectionSystem)
            logger.trace(f"Reflection input: {self.system.reflector.reflection_input}")
            logger.trace(f"Reflection output: {self.system.reflector.reflection_output}")

    def after_iteration(self, answer: Any, gt_answer: int | float | str, record: dict, pbar: tqdm) -> None:
        record['Answer_GT'] = gt_answer
        record['System_Finished'] = self.system.finished
        
        # Debug logging for system state
        sample_id = record.get('sample_id', 'unknown')
        user_id = record.get('user_id', 'unknown')
        logger.info(f"Sample {sample_id} (User {user_id}): system.finished={self.system.finished}, answer_type={type(answer)}, answer={answer}")
        
        # Track ground truth position in ranked list for SR/RR tasks
        if self.task in ['sr', 'rr'] and self.system.finished and isinstance(answer, list):
            try:
                if gt_answer in answer:
                    gt_position = answer.index(gt_answer) + 1  # 1-indexed position
                else:
                    gt_position = -1  # Not found in ranked list
                
                self.gt_positions.append({
                    'sample_id': sample_id,
                    'user_id': user_id,
                    'gt_item': gt_answer,
                    'position': gt_position,
                    'list_length': len(answer)
                })
            except Exception as e:
                logger.warning(f"Failed to track GT position for sample {sample_id}: {e}")
        
        # Track failed samples
        if not self.system.finished:
            sample_info = {
                'sample_id': record.get('sample_id', 'unknown'),
                'user_id': record.get('user_id', 'unknown')
            }
            self.failed_samples.append(sample_info)
        
        # Log sample completion status
        if self.system.finished:
            logger.info(f"Sample SUCCESS: {sample_id} (User {user_id})")
        else:
            logger.info(f"Sample FAILED: {sample_id} (User {user_id})")
        
        self.output_file.write(record)
        pbar.set_description(self.update_evaluation(answer, gt_answer))
        
        # **CHECKPOINT**: Log cumulative scores after each sample
        self._log_cumulative_scores(sample_id)

    def after_generate(self) -> None:
        self.output_file.close()
        logger.success("===================================Evaluation Report===================================")
        # Log valid answer statistics
        valid_percentage = (self.valid_count / self.total_count * 100) if self.total_count > 0 else 0
        logger.success(f"Valid Answers: {self.valid_count}/{self.total_count} samples ({valid_percentage:.1f}%)")
        
        # Log failed samples
        if self.failed_samples:
            logger.warning(f"Failed Samples ({len(self.failed_samples)}):")
            for failed_sample in self.failed_samples:
                logger.warning(f"  - Sample {failed_sample['sample_id']} (User {failed_sample['user_id']})")
        else:
            logger.success("All samples completed successfully!")
        
        self.metrics.report()
        
        # Log ground truth position summary for SR/RR tasks (only to file)
        if self.task in ['sr', 'rr'] and self.gt_positions:
            # Get the log file path from the task logger
            if hasattr(self, 'log_handler_id') and self.log_handler_id is not None:
                # Get handler info to find the log file path
                import sys
                from loguru._handler import Handler
                
                # Find the log file path from logger handlers
                log_file_path = None
                for handler_id, handler in logger._core.handlers.items():
                    if handler_id == self.log_handler_id:
                        # Extract file path from the handler
                        if hasattr(handler._sink, 'name'):
                            log_file_path = handler._sink.name
                        elif hasattr(handler._sink, '_file') and hasattr(handler._sink._file, 'name'):
                            log_file_path = handler._sink._file.name
                        break
                
                if log_file_path:
                    # Write directly to the log file
                    with open(log_file_path, 'a', encoding='utf-8') as log_file:
                        # Sort by position (ascending), with not-found (-1) at the end
                        sorted_positions = sorted(self.gt_positions, key=lambda x: (x['position'] == -1, x['position']))
                        
                        # Calculate statistics
                        found_count = sum(1 for p in self.gt_positions if p['position'] > 0)
                        not_found_count = sum(1 for p in self.gt_positions if p['position'] == -1)
                        
                        log_file.write("\n===================================Ground Truth Position Summary===================================\n")
                        
                        if found_count > 0:
                            positions_list = [p['position'] for p in self.gt_positions if p['position'] > 0]
                            avg_position = sum(positions_list) / len(positions_list)
                            log_file.write(f"Ground Truth Found: {found_count}/{len(self.gt_positions)} samples ({found_count/len(self.gt_positions)*100:.1f}%)\n")
                            log_file.write(f"Average Position (when found): {avg_position:.2f}\n")
                        
                        if not_found_count > 0:
                            log_file.write(f"Ground Truth Not Found: {not_found_count}/{len(self.gt_positions)} samples\n")
                        
                        log_file.write("\nDetailed Ground Truth Positions (sorted by position):\n")
                        log_file.write(f"{'Sample':<8} {'User':<8} {'GT Item':<10} {'Position':<10} {'List Length':<12}\n")
                        log_file.write("-" * 60 + "\n")
                        
                        for pos_info in sorted_positions:
                            position_str = str(pos_info['position']) if pos_info['position'] > 0 else "Not Found"
                            log_file.write(
                                f"{pos_info['sample_id']:<8} "
                                f"{pos_info['user_id']:<8} "
                                f"{pos_info['gt_item']:<10} "
                                f"{position_str:<10} "
                                f"{pos_info['list_length']:<12}\n"
                            )
        
        # Log reflection improvements if available
        if hasattr(self.system, 'reflection_all_reruns') and self.system.reflection_all_reruns:
            if hasattr(self, 'log_handler_id') and self.log_handler_id is not None:
                log_file_path = None
                for handler_id, handler in logger._core.handlers.items():
                    if handler_id == self.log_handler_id:
                        if hasattr(handler._sink, 'name'):
                            log_file_path = handler._sink.name
                        elif hasattr(handler._sink, '_file') and hasattr(handler._sink._file, 'name'):
                            log_file_path = handler._sink._file.name
                        break
                
                if log_file_path:
                    with open(log_file_path, 'a', encoding='utf-8') as log_file:
                        all_reruns = self.system.reflection_all_reruns
                        improvements = self.system.reflection_improvements if hasattr(self.system, 'reflection_improvements') else []
                        total_reflections_triggered = len(all_reruns)  # Use actual rerun count
                        
                        log_file.write("\n===================================Reflection Summary===================================\n")
                        log_file.write(f"Total reflection reruns triggered: {total_reflections_triggered}\n")
                        log_file.write(f"Reflection reruns with improvements: {len(improvements)}\n")
                        if total_reflections_triggered > 0:
                            log_file.write(f"Improvement rate: {len(improvements)}/{total_reflections_triggered} ({100*len(improvements)/total_reflections_triggered:.1f}%)\n")
                        
                        log_file.write("\n===================================Reflection Reruns Summary===================================\n")
                        log_file.write(f"Samples rerun by reflection: {len(all_reruns)}/{len(self.gt_positions)}\n")
                        
                        if all_reruns:
                            # Categorize reruns by feedback type
                            planner_only = [r for r in all_reruns if r.get('feedback_type') == 'planner']
                            solver_only = [r for r in all_reruns if r.get('feedback_type') == 'solver']
                            both_agents = [r for r in all_reruns if r.get('feedback_type') == 'both']
                            
                            # Calculate improvements by category
                            def count_improvements(reruns):
                                return sum(1 for r in reruns if r['position_after'] < r['position_before'] and r['position_before'] > 0)
                            
                            log_file.write("\n📊 BREAKDOWN BY FEEDBACK TYPE:\n")
                            log_file.write(f"  • Planner only (full rerun):        {len(planner_only)} samples | {count_improvements(planner_only)} improved\n")
                            log_file.write(f"  • Solver only (reranking):          {len(solver_only)} samples | {count_improvements(solver_only)} improved\n")
                            log_file.write(f"  • Both agents (full rerun):         {len(both_agents)} samples | {count_improvements(both_agents)} improved\n")
                            
                            log_file.write("\n📈 DETAILED BREAKDOWN:\n")
                            log_file.write("All Reflection Reruns (sorted by improvement, positive = improved, negative = worsened):\n")
                            log_file.write(f"{'Sample':<8} {'User':<8} {'GT Item':<10} {'Before':<8} {'After':<8} {'Δ':<8} {'Feedback Type':<18}\n")
                            log_file.write("-" * 80 + "\n")
                            
                            # Sort by improvement (most improved first, then least worsened)
                            sorted_reruns = sorted(all_reruns, key=lambda x: x['position_before'] - x['position_after'], reverse=True)
                            
                            for rerun_info in sorted_reruns:
                                improvement = rerun_info['position_before'] - rerun_info['position_after']
                                feedback_type = rerun_info.get('feedback_type', 'unknown')
                                
                                # Format improvement with emoji
                                if improvement > 0:
                                    improvement_str = f"✅ +{improvement}"
                                elif improvement < 0:
                                    improvement_str = f"⚠️  {improvement}"
                                else:
                                    improvement_str = "→ 0"
                                
                                # Format feedback type with emoji
                                if feedback_type == 'planner':
                                    feedback_str = "Planner (Full)"
                                elif feedback_type == 'solver':
                                    feedback_str = "Solver (Rerank)"
                                elif feedback_type == 'both':
                                    feedback_str = "Both (Full)"
                                else:
                                    feedback_str = "Unknown"
                                
                                log_file.write(
                                    f"{rerun_info['sample_idx']:<8} "
                                    f"{rerun_info['user_id']:<8} "
                                    f"{rerun_info['gt_item']:<10} "
                                    f"{rerun_info['position_before']:<8} "
                                    f"{rerun_info['position_after']:<8} "
                                    f"{improvement_str:<8} "
                                    f"{feedback_str:<18}\n"
                                )
                            
                            # Add summary statistics at the end
                            log_file.write("\n" + "="*80 + "\n")
                            log_file.write("📊 REFLECTION STATISTICS:\n")
                            improved_count = len(improvements)
                            worsened_count = sum(1 for r in all_reruns if r['position_after'] > r['position_before'] and r['position_before'] > 0)
                            unchanged_count = sum(1 for r in all_reruns if r['position_after'] == r['position_before'])
                            
                            log_file.write(f"  • Improved:   {improved_count:3d} samples (↓ position)\n")
                            log_file.write(f"  • Worsened:   {worsened_count:3d} samples (↑ position)\n")
                            log_file.write(f"  • Unchanged:  {unchanged_count:3d} samples (= position)\n")
                            log_file.write(f"  • Success rate: {100*improved_count/len(all_reruns):.1f}%\n")
                            
                            # Calculate average improvement for improved samples
                            if improved_count > 0:
                                avg_improvement = sum(r['position_before'] - r['position_after'] for r in all_reruns if r['position_after'] < r['position_before'] and r['position_before'] > 0) / improved_count
                                log_file.write(f"  • Average improvement (for improved samples): {avg_improvement:.2f} positions\n")
                            
                            log_file.write("="*80 + "\n")

        

    def run(self, steps: int, topks: list[int], *args, **kwargs):
        assert kwargs['task'] in ['rp', 'sr', 'rr'], "Only support rating (rp) and ranking (sr/rr) tasks."
        self.steps = steps
        self.topks = topks
        super().run(*args, **kwargs)

if __name__ == '__main__':
    EvaluateTask().launch()
