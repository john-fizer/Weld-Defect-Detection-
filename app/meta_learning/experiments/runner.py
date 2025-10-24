"""
Experiment Runner

Orchestrates experiments comparing meta-prompting vs fine-tuning
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

from ..acla.curriculum_agent import AdaptiveCurriculumAgent, CurriculumConfig
from ..clrs.reinforcement_system import ClosedLoopSystem
from .evaluator import ModelEvaluator
from .comparator import ExperimentComparator


@dataclass
class ExperimentConfig:
    """Configuration for an experiment"""
    name: str
    description: str
    dataset_name: str
    approach: str  # 'meta_prompting' or 'static_baseline'
    num_iterations: int = 10
    sample_size: int = 100
    metadata: Dict[str, Any] = None


@dataclass
class ExperimentResult:
    """Results from an experiment"""
    config: ExperimentConfig
    final_performance: Dict[str, float]
    performance_history: List[Dict[str, float]]
    best_performance: Dict[str, float]
    improvement: float
    convergence_iteration: int
    timestamp: datetime
    metadata: Dict[str, Any]


class ExperimentRunner:
    """
    Run and coordinate meta-learning experiments

    Key Research Questions:
    1. Can meta-prompting outperform static fine-tuning?
    2. How many iterations to convergence?
    3. Which evolution strategies work best?
    4. How does performance scale with dataset size?
    """

    def __init__(
        self,
        save_path: Optional[Path] = None,
        llm_client: Optional[Any] = None
    ):
        self.save_path = save_path or Path("./data/meta_learning/experiments")
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.llm_client = llm_client

        self.evaluator = ModelEvaluator()
        self.comparator = ExperimentComparator()

        self.experiments: List[ExperimentResult] = []

    async def run_meta_prompting_experiment(
        self,
        config: ExperimentConfig,
        dataset_loader,
        initial_prompt: str
    ) -> ExperimentResult:
        """
        Run meta-prompting experiment with ACLA

        Args:
            config: Experiment configuration
            dataset_loader: Dataset loader function
            initial_prompt: Starting prompt

        Returns:
            Experiment results
        """
        print(f"\n{'='*70}")
        print(f"RUNNING META-PROMPTING EXPERIMENT: {config.name}")
        print(f"{'='*70}\n")

        # Create ACLA configuration
        acla_config = CurriculumConfig(
            initial_prompt=initial_prompt,
            dataset_name=config.dataset_name,
            max_iterations=config.num_iterations,
            llm_provider="anthropic",
            model_name="claude-3-5-sonnet-20241022"
        )

        # Initialize ACLA
        agent = AdaptiveCurriculumAgent(
            config=acla_config,
            llm_client=self.llm_client,
            save_path=self.save_path / "acla" / config.name
        )

        # Create evaluation function
        async def evaluation_fn(prompt: str, samples: List[Dict]) -> Dict[str, float]:
            return await self.evaluator.evaluate_prompt(
                prompt=prompt,
                samples=samples,
                llm_client=self.llm_client,
                dataset_name=config.dataset_name
            )

        # Run curriculum learning
        results = await agent.run_curriculum(
            dataset_loader=lambda n: dataset_loader(n or config.sample_size),
            evaluation_fn=evaluation_fn,
            num_samples=config.sample_size
        )

        # Extract results
        experiment_result = ExperimentResult(
            config=config,
            final_performance=results['iterations'][-1]['performance_metrics'],
            performance_history=[it['performance_metrics'] for it in results['iterations']],
            best_performance={'accuracy': results['best_performance']},
            improvement=results['total_improvement'],
            convergence_iteration=results['convergence_iteration'],
            timestamp=datetime.now(),
            metadata={
                'approach': 'meta_prompting',
                'best_prompt': results['best_prompt'],
                'strategy_analysis': results['strategy_analysis']
            }
        )

        self.experiments.append(experiment_result)
        self._save_experiment(experiment_result)

        return experiment_result

    async def run_static_baseline_experiment(
        self,
        config: ExperimentConfig,
        dataset_loader,
        prompt: str
    ) -> ExperimentResult:
        """
        Run baseline experiment with static prompt (no adaptation)

        Args:
            config: Experiment configuration
            dataset_loader: Dataset loader function
            prompt: Static prompt to use

        Returns:
            Experiment results
        """
        print(f"\n{'='*70}")
        print(f"RUNNING STATIC BASELINE EXPERIMENT: {config.name}")
        print(f"{'='*70}\n")

        # Load dataset
        samples = dataset_loader(config.sample_size)

        # Evaluate static prompt multiple times to simulate iterations
        performance_history = []

        for iteration in range(config.num_iterations):
            print(f"\nIteration {iteration + 1}/{config.num_iterations}")

            # Evaluate (same prompt each time)
            metrics = await self.evaluator.evaluate_prompt(
                prompt=prompt,
                samples=samples,
                llm_client=self.llm_client,
                dataset_name=config.dataset_name
            )

            performance_history.append(metrics)

            print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")

        # Calculate results
        final_performance = performance_history[-1]
        best_performance = max(
            performance_history,
            key=lambda m: m.get('accuracy', 0)
        )
        improvement = 0.0  # No improvement expected for static baseline

        experiment_result = ExperimentResult(
            config=config,
            final_performance=final_performance,
            performance_history=performance_history,
            best_performance=best_performance,
            improvement=improvement,
            convergence_iteration=0,
            timestamp=datetime.now(),
            metadata={
                'approach': 'static_baseline',
                'prompt': prompt
            }
        )

        self.experiments.append(experiment_result)
        self._save_experiment(experiment_result)

        return experiment_result

    async def run_comparison_study(
        self,
        dataset_name: str,
        dataset_loader,
        initial_prompt: str,
        num_iterations: int = 10,
        sample_size: int = 100
    ) -> Dict[str, Any]:
        """
        Run complete comparison study: meta-prompting vs static baseline

        Args:
            dataset_name: Name of dataset
            dataset_loader: Dataset loader function
            initial_prompt: Starting prompt
            num_iterations: Number of iterations
            sample_size: Sample size per iteration

        Returns:
            Comparison results
        """
        print(f"\n{'='*70}")
        print(f"META-PROMPTING VS FINE-TUNING COMPARISON STUDY")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}\n")

        # Configure experiments
        meta_config = ExperimentConfig(
            name=f"{dataset_name}_meta_prompting",
            description="Meta-prompting with adaptive curriculum learning",
            dataset_name=dataset_name,
            approach="meta_prompting",
            num_iterations=num_iterations,
            sample_size=sample_size
        )

        baseline_config = ExperimentConfig(
            name=f"{dataset_name}_static_baseline",
            description="Static prompt baseline (no adaptation)",
            dataset_name=dataset_name,
            approach="static_baseline",
            num_iterations=num_iterations,
            sample_size=sample_size
        )

        # Run both experiments
        print("\n1. Running Meta-Prompting Experiment...")
        meta_result = await self.run_meta_prompting_experiment(
            meta_config,
            dataset_loader,
            initial_prompt
        )

        print("\n2. Running Static Baseline Experiment...")
        baseline_result = await self.run_static_baseline_experiment(
            baseline_config,
            dataset_loader,
            initial_prompt
        )

        # Compare results
        comparison = self.comparator.compare_experiments(
            meta_result,
            baseline_result
        )

        # Save comparison
        self._save_comparison(comparison, dataset_name)

        # Print summary
        self._print_comparison_summary(comparison)

        return comparison

    def _save_experiment(self, result: ExperimentResult):
        """Save experiment results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.config.name}_{timestamp}.json"
        filepath = self.save_path / filename

        # Convert to dict
        data = {
            'config': asdict(result.config),
            'final_performance': result.final_performance,
            'performance_history': result.performance_history,
            'best_performance': result.best_performance,
            'improvement': result.improvement,
            'convergence_iteration': result.convergence_iteration,
            'timestamp': result.timestamp.isoformat(),
            'metadata': result.metadata
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"\nExperiment results saved: {filepath}")

    def _save_comparison(self, comparison: Dict[str, Any], dataset_name: str):
        """Save comparison results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_{dataset_name}_{timestamp}.json"
        filepath = self.save_path / filename

        with open(filepath, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)

        print(f"\nComparison results saved: {filepath}")

    def _print_comparison_summary(self, comparison: Dict[str, Any]):
        """Print comparison summary"""
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")

        print(f"\nFinal Performance:")
        print(f"  Meta-Prompting: {comparison['final_performance']['meta_prompting']:.4f}")
        print(f"  Static Baseline: {comparison['final_performance']['baseline']:.4f}")
        print(f"  Difference: {comparison['final_performance']['difference']:.4f}")

        print(f"\nBest Performance:")
        print(f"  Meta-Prompting: {comparison['best_performance']['meta_prompting']:.4f}")
        print(f"  Static Baseline: {comparison['best_performance']['baseline']:.4f}")
        print(f"  Difference: {comparison['best_performance']['difference']:.4f}")

        print(f"\nImprovement:")
        print(f"  Meta-Prompting: {comparison['improvement']['meta_prompting']:.4f}")
        print(f"  Static Baseline: {comparison['improvement']['baseline']:.4f}")

        print(f"\nConclusion:")
        if comparison['meta_prompting_wins']:
            print("  ✓ Meta-prompting OUTPERFORMS static baseline")
            print(f"    Win margin: {comparison['win_margin']:.4f}")
        else:
            print("  ✗ Meta-prompting does NOT outperform static baseline")
            print(f"    Performance gap: {comparison['win_margin']:.4f}")

        print(f"\n{'='*70}\n")

    async def run_clrs_experiment(
        self,
        config: ExperimentConfig,
        generate_fn: Callable,
        input_samples: List[str],
        feedback_fn: Callable,
        num_cycles: int = 5
    ) -> ExperimentResult:
        """
        Run Closed-Loop Reinforcement System experiment

        Args:
            config: Experiment configuration
            generate_fn: Function to generate outputs
            input_samples: Input samples
            feedback_fn: Function to generate feedback
            num_cycles: Number of training cycles

        Returns:
            Experiment results
        """
        print(f"\n{'='*70}")
        print(f"RUNNING CLRS EXPERIMENT: {config.name}")
        print(f"{'='*70}\n")

        # Initialize CLRS
        clrs = ClosedLoopSystem(
            save_path=self.save_path / "clrs" / config.name,
            cycle_size=len(input_samples)
        )

        # Run simulation
        summary = clrs.simulate_feedback_loop(
            generate_fn=generate_fn,
            input_samples=input_samples,
            feedback_fn=feedback_fn,
            num_cycles=num_cycles
        )

        # Create result
        experiment_result = ExperimentResult(
            config=config,
            final_performance={
                'drift': summary['drift']['current'],
                'alignment': summary['alignment']['current'],
                'coherence': summary['coherence']['current']
            },
            performance_history=[
                {
                    'drift': c.drift_score,
                    'alignment': c.alignment_score,
                    'coherence': c.coherence_score
                }
                for c in clrs.training_cycles
            ],
            best_performance={
                'drift': summary['drift']['max'],
                'alignment': summary['alignment']['max'],
                'coherence': summary['coherence']['max']
            },
            improvement=summary['alignment']['current'] - summary['alignment']['average'],
            convergence_iteration=summary['total_cycles'],
            timestamp=datetime.now(),
            metadata={
                'approach': 'clrs',
                'summary': summary
            }
        )

        self.experiments.append(experiment_result)
        self._save_experiment(experiment_result)

        return experiment_result

    def get_all_experiments(self) -> List[ExperimentResult]:
        """Get all experiments run"""
        return self.experiments

    def load_experiment(self, filepath: Path) -> ExperimentResult:
        """Load a saved experiment"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Reconstruct result (simplified)
        config = ExperimentConfig(**data['config'])

        result = ExperimentResult(
            config=config,
            final_performance=data['final_performance'],
            performance_history=data['performance_history'],
            best_performance=data['best_performance'],
            improvement=data['improvement'],
            convergence_iteration=data['convergence_iteration'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data['metadata']
        )

        return result
