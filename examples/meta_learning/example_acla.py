"""
Example: Adaptive Curriculum Learning Agent (ACLA)

Demonstrates meta-prompting with self-optimizing prompts on CommonsenseQA dataset

Research Question: Can meta-prompting outperform static fine-tuning?
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.meta_learning.acla import AdaptiveCurriculumAgent, CurriculumConfig
from app.meta_learning.datasets import CommonsenseQALoader
from app.meta_learning.experiments import ModelEvaluator
from app.meta_learning.utils import MetaLearningVisualizer, ExperimentLogger


async def main():
    """Run ACLA experiment on CommonsenseQA"""

    print("="*70)
    print("ADAPTIVE CURRICULUM LEARNING AGENT - CommonsenseQA")
    print("="*70)

    # Initialize logger
    logger = ExperimentLogger(experiment_name="acla_commonsense_qa")

    # Load dataset
    print("\nLoading CommonsenseQA dataset...")
    dataset = CommonsenseQALoader()
    dataset.load_data()

    stats = dataset.get_statistics()
    print(f"Dataset loaded: {stats['total_samples']} samples")
    print(f"  Train: {stats['train_samples']}")
    print(f"  Test: {stats['test_samples']}")

    # Initial prompt
    initial_prompt = dataset.create_prompt_template(include_examples=True)

    print("\nInitial Prompt:")
    print("-"*70)
    print(initial_prompt)
    print("-"*70)

    # Configure ACLA
    config = CurriculumConfig(
        initial_prompt=initial_prompt,
        dataset_name="commonsense_qa",
        max_iterations=5,  # Small number for demo
        min_performance_threshold=0.8,
        improvement_threshold=0.03,
        evolution_strategies=[
            'performance_based',
            'error_analysis',
            'chain_of_thought',
            'few_shot_optimization'
        ]
    )

    logger.log_config({
        'dataset': 'commonsense_qa',
        'max_iterations': config.max_iterations,
        'target_performance': config.min_performance_threshold
    })

    # Initialize agent (without LLM client for demo)
    agent = AdaptiveCurriculumAgent(
        config=config,
        llm_client=None,  # Set to actual client if available
        save_path=Path("./data/meta_learning/acla/commonsense_qa_demo")
    )

    # Create evaluator
    evaluator = ModelEvaluator()

    # Evaluation function
    async def evaluate_prompt(prompt: str, samples):
        """Evaluate prompt on samples"""
        return await evaluator.evaluate_prompt(
            prompt=prompt,
            samples=samples,
            llm_client=None,  # Set to actual client if available
            dataset_name="commonsense_qa"
        )

    # Run curriculum learning
    print("\nStarting curriculum learning...")
    print("NOTE: This demo uses simulated evaluation.")
    print("For real experiments, provide an LLM client (Anthropic or OpenAI).\n")

    try:
        results = await agent.run_curriculum(
            dataset_loader=lambda n: dataset.get_samples(n, split='test'),
            evaluation_fn=evaluate_prompt,
            num_samples=20  # Small sample for demo
        )

        # Log results
        logger.log_summary({
            'best_performance': results['best_performance'],
            'total_improvement': results['total_improvement'],
            'convergence_iteration': results['convergence_iteration']
        })

        # Visualize
        visualizer = MetaLearningVisualizer()
        visualizer.plot_curriculum_learning(
            performance_history=results['performance_history'],
            title="ACLA on CommonsenseQA",
            save_name="acla_commonsense_qa.png"
        )

        # Print final results
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(f"Best Performance: {results['best_performance']:.4f}")
        print(f"Total Improvement: {results['total_improvement']:.4f}")
        print(f"Iterations to Convergence: {results['convergence_iteration']}")

        print("\nBest Prompt:")
        print("-"*70)
        print(results['best_prompt'])
        print("-"*70)

        logger.close()

    except Exception as e:
        logger.log_error(e, context="curriculum_learning")
        raise


if __name__ == "__main__":
    asyncio.run(main())
