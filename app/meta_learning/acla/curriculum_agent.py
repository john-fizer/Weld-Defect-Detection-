"""
Adaptive Curriculum Learning Agent

Self-optimizing agent that evolves its prompts based on performance feedback
"""

import json
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import numpy as np
from dataclasses import dataclass, asdict

from .prompt_evolver import PromptEvolver
from .performance_tracker import PerformanceTracker


@dataclass
class PromptIteration:
    """Record of a prompt evolution iteration"""
    iteration: int
    timestamp: datetime
    prompt: str
    performance_metrics: Dict[str, float]
    dataset: str
    sample_size: int
    evolution_strategy: str
    metadata: Dict[str, Any]


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning"""
    initial_prompt: str
    dataset_name: str
    max_iterations: int = 10
    min_performance_threshold: float = 0.7
    improvement_threshold: float = 0.05  # Min improvement to continue
    evolution_strategies: List[str] = None
    llm_provider: str = "anthropic"  # "anthropic" or "openai"
    model_name: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.7

    def __post_init__(self):
        if self.evolution_strategies is None:
            self.evolution_strategies = [
                'performance_based',
                'error_analysis',
                'ablation',
                'chain_of_thought',
                'few_shot_optimization'
            ]


class AdaptiveCurriculumAgent:
    """
    An LLM agent that iteratively improves its own prompts through:
    1. Testing current prompt on dataset
    2. Analyzing performance and errors
    3. Generating improved prompt variations
    4. Selecting best performing prompt
    5. Repeating until convergence or max iterations

    Research Focus:
    - Does meta-prompting outperform static fine-tuning?
    - What evolution strategies work best?
    - How many iterations to convergence?
    - Emergent prompt patterns and structures
    """

    def __init__(
        self,
        config: CurriculumConfig,
        llm_client: Optional[Any] = None,
        save_path: Optional[Path] = None
    ):
        self.config = config
        self.llm_client = llm_client
        self.save_path = save_path or Path("./data/meta_learning/acla")
        self.save_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.evolver = PromptEvolver(
            llm_client=llm_client,
            provider=config.llm_provider,
            model=config.model_name
        )
        self.tracker = PerformanceTracker()

        # State
        self.current_prompt = config.initial_prompt
        self.iterations: List[PromptIteration] = []
        self.best_prompt: Optional[str] = None
        self.best_performance: float = 0.0

    async def run_curriculum(
        self,
        dataset_loader,
        evaluation_fn,
        num_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run the full adaptive curriculum learning process

        Args:
            dataset_loader: Function that returns dataset samples
            evaluation_fn: Function that evaluates prompt performance
                          Signature: async fn(prompt, samples) -> Dict[metrics]
            num_samples: Number of samples to evaluate per iteration

        Returns:
            Dictionary with final results and learning history
        """
        print(f"\n{'='*70}")
        print(f"ADAPTIVE CURRICULUM LEARNING AGENT - Starting")
        print(f"{'='*70}")
        print(f"Dataset: {self.config.dataset_name}")
        print(f"Max Iterations: {self.config.max_iterations}")
        print(f"Target Performance: {self.config.min_performance_threshold}")
        print(f"{'='*70}\n")

        # Load dataset
        dataset_samples = dataset_loader(num_samples)

        for iteration in range(self.config.max_iterations):
            print(f"\n{'─'*70}")
            print(f"ITERATION {iteration + 1}/{self.config.max_iterations}")
            print(f"{'─'*70}")

            # Evaluate current prompt
            print(f"\nEvaluating current prompt...")
            metrics = await evaluation_fn(self.current_prompt, dataset_samples)

            # Track performance
            self.tracker.record_iteration(iteration, metrics)

            # Create iteration record
            iteration_record = PromptIteration(
                iteration=iteration + 1,
                timestamp=datetime.now(),
                prompt=self.current_prompt,
                performance_metrics=metrics,
                dataset=self.config.dataset_name,
                sample_size=len(dataset_samples),
                evolution_strategy=self._select_evolution_strategy(iteration),
                metadata={
                    'config': asdict(self.config)
                }
            )
            self.iterations.append(iteration_record)

            # Display metrics
            self._display_metrics(metrics, iteration + 1)

            # Update best if improved
            current_score = metrics.get('accuracy', metrics.get('f1', 0))
            if current_score > self.best_performance:
                self.best_performance = current_score
                self.best_prompt = self.current_prompt
                print(f"\n🎯 NEW BEST PERFORMANCE: {current_score:.4f}")

            # Check convergence
            if self._check_convergence(iteration):
                print(f"\n✓ Convergence achieved at iteration {iteration + 1}")
                break

            # Check if target met
            if current_score >= self.config.min_performance_threshold:
                print(f"\n✓ Target performance threshold reached!")
                break

            # Evolve prompt for next iteration
            print(f"\nEvolving prompt...")
            evolution_strategy = self._select_evolution_strategy(iteration)

            new_prompt = await self.evolver.evolve_prompt(
                current_prompt=self.current_prompt,
                performance_metrics=metrics,
                error_examples=self._extract_error_examples(iteration),
                strategy=evolution_strategy,
                iteration=iteration + 1
            )

            self.current_prompt = new_prompt

            # Save checkpoint
            self._save_checkpoint(iteration + 1)

        # Final results
        results = self._compile_results()
        self._save_final_results(results)

        print(f"\n{'='*70}")
        print(f"CURRICULUM LEARNING COMPLETE")
        print(f"{'='*70}")
        print(f"Best Performance: {self.best_performance:.4f}")
        print(f"Total Iterations: {len(self.iterations)}")
        print(f"Improvement: {self._calculate_total_improvement():.4f}")
        print(f"{'='*70}\n")

        return results

    def _select_evolution_strategy(self, iteration: int) -> str:
        """Select evolution strategy based on iteration"""
        strategies = self.config.evolution_strategies

        # Cycle through strategies or use adaptive selection
        if iteration < len(strategies):
            return strategies[iteration]

        # After first cycle, use performance-based selection
        return self._adaptive_strategy_selection()

    def _adaptive_strategy_selection(self) -> str:
        """Select strategy based on what worked best historically"""
        if len(self.iterations) < 2:
            return 'performance_based'

        # Analyze which strategies led to biggest improvements
        strategy_improvements = {}

        for i in range(1, len(self.iterations)):
            prev_score = self.iterations[i-1].performance_metrics.get('accuracy', 0)
            curr_score = self.iterations[i].performance_metrics.get('accuracy', 0)
            improvement = curr_score - prev_score

            strategy = self.iterations[i].evolution_strategy
            if strategy not in strategy_improvements:
                strategy_improvements[strategy] = []
            strategy_improvements[strategy].append(improvement)

        # Select strategy with best average improvement
        best_strategy = max(
            strategy_improvements.keys(),
            key=lambda s: np.mean(strategy_improvements[s])
        )

        return best_strategy

    def _check_convergence(self, iteration: int) -> bool:
        """Check if learning has converged"""
        if iteration < 2:
            return False

        # Check last 3 iterations for improvement
        window_size = min(3, iteration + 1)
        recent_scores = [
            self.iterations[i].performance_metrics.get('accuracy', 0)
            for i in range(len(self.iterations) - window_size, len(self.iterations))
        ]

        # Calculate improvement trend
        improvements = [
            recent_scores[i] - recent_scores[i-1]
            for i in range(1, len(recent_scores))
        ]

        # Converged if improvements are all below threshold
        max_improvement = max(improvements) if improvements else 0

        return max_improvement < self.config.improvement_threshold

    def _extract_error_examples(self, iteration: int) -> List[Dict]:
        """Extract examples of errors from recent iterations"""
        # This would be populated by the evaluation function
        # For now, return empty list (implementation depends on dataset)
        return []

    def _display_metrics(self, metrics: Dict[str, float], iteration: int):
        """Display performance metrics"""
        print(f"\nIteration {iteration} Results:")
        print(f"{'─'*40}")
        for metric_name, value in metrics.items():
            print(f"  {metric_name:20s}: {value:.4f}")
        print(f"{'─'*40}")

    def _calculate_total_improvement(self) -> float:
        """Calculate improvement from first to best iteration"""
        if not self.iterations:
            return 0.0

        first_score = self.iterations[0].performance_metrics.get('accuracy', 0)
        return self.best_performance - first_score

    def _compile_results(self) -> Dict[str, Any]:
        """Compile final results"""
        return {
            'config': asdict(self.config),
            'iterations': [asdict(it) for it in self.iterations],
            'best_prompt': self.best_prompt,
            'best_performance': self.best_performance,
            'total_improvement': self._calculate_total_improvement(),
            'convergence_iteration': len(self.iterations),
            'performance_history': self.tracker.get_history(),
            'strategy_analysis': self._analyze_strategies(),
            'timestamp': datetime.now().isoformat()
        }

    def _analyze_strategies(self) -> Dict[str, Any]:
        """Analyze which evolution strategies worked best"""
        strategy_stats = {}

        for i, iteration in enumerate(self.iterations):
            strategy = iteration.evolution_strategy
            score = iteration.performance_metrics.get('accuracy', 0)

            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'count': 0,
                    'scores': [],
                    'improvements': []
                }

            strategy_stats[strategy]['count'] += 1
            strategy_stats[strategy]['scores'].append(score)

            if i > 0:
                prev_score = self.iterations[i-1].performance_metrics.get('accuracy', 0)
                improvement = score - prev_score
                strategy_stats[strategy]['improvements'].append(improvement)

        # Calculate aggregates
        for strategy, stats in strategy_stats.items():
            stats['avg_score'] = np.mean(stats['scores'])
            stats['avg_improvement'] = np.mean(stats['improvements']) if stats['improvements'] else 0
            stats['success_rate'] = sum(1 for imp in stats['improvements'] if imp > 0) / len(stats['improvements']) if stats['improvements'] else 0

        return strategy_stats

    def _save_checkpoint(self, iteration: int):
        """Save checkpoint after each iteration"""
        checkpoint_path = self.save_path / f"checkpoint_iter_{iteration}.json"

        checkpoint_data = {
            'iteration': iteration,
            'current_prompt': self.current_prompt,
            'best_prompt': self.best_prompt,
            'best_performance': self.best_performance,
            'timestamp': datetime.now().isoformat()
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

    def _save_final_results(self, results: Dict[str, Any]):
        """Save final results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.save_path / f"acla_results_{timestamp}.json"

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {results_path}")

    def get_best_prompt(self) -> str:
        """Get the best performing prompt"""
        return self.best_prompt or self.current_prompt

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance across iterations"""
        return {
            'total_iterations': len(self.iterations),
            'best_performance': self.best_performance,
            'initial_performance': self.iterations[0].performance_metrics.get('accuracy', 0) if self.iterations else 0,
            'improvement': self._calculate_total_improvement(),
            'convergence_rate': len(self.iterations) / self.config.max_iterations,
            'strategy_analysis': self._analyze_strategies()
        }
