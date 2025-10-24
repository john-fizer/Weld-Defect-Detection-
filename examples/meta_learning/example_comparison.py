"""
Example: Meta-Prompting vs Static Baseline Comparison

Demonstrates the full comparison study to answer:
"Can meta-prompting outperform static fine-tuning?"

This is the key research experiment for the meta-learning framework.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.meta_learning.datasets import Sentiment140Loader
from app.meta_learning.experiments import ExperimentRunner
from app.meta_learning.utils import MetaLearningVisualizer, ExperimentLogger


async def main():
    """Run comparison study: meta-prompting vs static baseline"""

    print("="*70)
    print("META-PROMPTING VS STATIC BASELINE COMPARISON")
    print("="*70)

    # Initialize logger
    logger = ExperimentLogger(experiment_name="comparison_sentiment140")

    # Load dataset
    print("\nLoading Sentiment140 dataset...")
    dataset = Sentiment140Loader()
    dataset.load_data()
    dataset.balance_dataset(max_per_class=50)  # Balance for fair comparison

    stats = dataset.get_statistics()
    print(f"Dataset loaded: {stats['total_samples']} samples")
    print(f"  Distribution: {stats['sentiment_distribution']}")

    # Initial prompt
    initial_prompt = dataset.create_prompt_template(include_examples=True)

    print("\nInitial Prompt:")
    print("-"*70)
    print(initial_prompt)
    print("-"*70)

    # Initialize experiment runner
    runner = ExperimentRunner(
        save_path=Path("./data/meta_learning/experiments/comparison_demo"),
        llm_client=None  # Set to actual client if available
    )

    # Configure experiment
    logger.log_config({
        'dataset': 'sentiment140',
        'num_iterations': 5,
        'sample_size': 30,
        'comparison': 'meta_prompting_vs_static_baseline'
    })

    # Run comparison study
    print("\nRunning comparison study...")
    print("NOTE: This demo uses simulated evaluation.")
    print("For real experiments, provide an LLM client.\n")

    try:
        comparison = await runner.run_comparison_study(
            dataset_name="sentiment140",
            dataset_loader=lambda n: dataset.get_samples(n, split='test'),
            initial_prompt=initial_prompt,
            num_iterations=5,
            sample_size=30
        )

        # Log comparison
        logger.log_comparison(comparison)

        # Visualize comparison
        visualizer = MetaLearningVisualizer()

        # Get experiment results
        experiments = runner.get_all_experiments()
        meta_exp = next((e for e in experiments if e.config.approach == 'meta_prompting'), None)
        baseline_exp = next((e for e in experiments if e.config.approach == 'static_baseline'), None)

        if meta_exp and baseline_exp:
            visualizer.plot_comparison(
                meta_prompting_history=meta_exp.performance_history,
                baseline_history=baseline_exp.performance_history,
                metric='accuracy',
                title="Meta-Prompting vs Static Baseline on Sentiment140",
                save_name="comparison_sentiment140.png"
            )

        # Print detailed comparison
        print("\n" + "="*70)
        print("DETAILED COMPARISON RESULTS")
        print("="*70)

        print("\n1. FINAL PERFORMANCE")
        print("-"*40)
        print(f"  Meta-Prompting:  {comparison['final_performance']['meta_prompting']:.4f}")
        print(f"  Static Baseline: {comparison['final_performance']['baseline']:.4f}")
        print(f"  Difference:      {comparison['final_performance']['difference']:.4f}")
        print(f"  Relative Gain:   {comparison['final_performance']['relative_improvement']*100:.2f}%")

        print("\n2. BEST PERFORMANCE")
        print("-"*40)
        print(f"  Meta-Prompting:  {comparison['best_performance']['meta_prompting']:.4f}")
        print(f"  Static Baseline: {comparison['best_performance']['baseline']:.4f}")
        print(f"  Difference:      {comparison['best_performance']['difference']:.4f}")

        print("\n3. IMPROVEMENT")
        print("-"*40)
        print(f"  Meta-Prompting:  {comparison['improvement']['meta_prompting']:.4f}")
        print(f"  Static Baseline: {comparison['improvement']['baseline']:.4f}")
        print(f"  Difference:      {comparison['improvement']['difference']:.4f}")

        print("\n4. LEARNING EFFICIENCY")
        print("-"*40)
        print(f"  Meta-Prompting Avg Improvement: {comparison['learning_efficiency']['meta_prompting_avg_improvement']:.4f}")
        print(f"  Baseline Avg Improvement:       {comparison['learning_efficiency']['baseline_avg_improvement']:.4f}")
        print(f"  More Efficient: {comparison['learning_efficiency']['meta_prompting_more_efficient']}")

        print("\n5. STATISTICAL SIGNIFICANCE")
        print("-"*40)
        print(f"  Significant: {comparison['statistical_significance']['significant']}")
        print(f"  Effect Size: {comparison['statistical_significance']['effect_size']:.4f}")

        print("\n6. CONCLUSION")
        print("-"*40)
        print(f"  {comparison['conclusion']}")

        if comparison['meta_prompting_wins']:
            print("\n  ✓ META-PROMPTING WINS!")
            print(f"    Win margin: {comparison['win_margin']:.4f}")
        else:
            print("\n  ✗ Static baseline performed better")

        logger.close()

    except Exception as e:
        logger.log_error(e, context="comparison_study")
        raise


if __name__ == "__main__":
    asyncio.run(main())
