"""
Example: Closed-Loop Reinforcement System (CLRS)

Demonstrates feedback-driven model training with drift/alignment/coherence monitoring

Studies: drift, alignment, and emergent coherence over multiple feedback cycles
"""

import sys
from pathlib import Path
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.meta_learning.clrs import ClosedLoopSystem
from app.meta_learning.utils import MetaLearningVisualizer, ExperimentLogger


def main():
    """Run CLRS experiment"""

    print("="*70)
    print("CLOSED-LOOP REINFORCEMENT SYSTEM")
    print("="*70)

    # Initialize logger
    logger = ExperimentLogger(experiment_name="clrs_demo")

    # Initialize CLRS
    clrs = ClosedLoopSystem(
        save_path=Path("./data/meta_learning/clrs/demo"),
        cycle_size=20,  # 20 samples per training cycle
        drift_threshold=0.3,
        alignment_threshold=0.7
    )

    logger.log_config({
        'cycle_size': 20,
        'drift_threshold': 0.3,
        'alignment_threshold': 0.7
    })

    # Sample inputs for demonstration
    input_samples = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms",
        "Write a haiku about spring",
        "How does photosynthesis work?",
        "What are the benefits of exercise?",
        "Describe the water cycle",
        "What is machine learning?",
        "How do vaccines work?",
        "Explain gravity",
        "What causes seasons?",
        "How do computers process information?",
        "What is climate change?",
        "Explain the theory of relativity",
        "How does the internet work?",
        "What is DNA?",
        "How do airplanes fly?",
        "What is blockchain?",
        "Explain photosynthesis",
        "How does memory work?",
        "What is evolution?",
    ] * 5  # Repeat for multiple cycles

    # Simulated generation function
    def generate_output(input_text: str) -> str:
        """Simulate model output generation"""
        # In real implementation, this would call an actual model
        responses = [
            f"Here's an answer to your question about {input_text[:30]}...",
            f"Let me explain: {input_text[:30]}...",
            f"The answer is related to {input_text[:30]}...",
        ]
        return random.choice(responses)

    # Simulated feedback function
    def generate_feedback(input_text: str, output_text: str) -> float:
        """Simulate user feedback (0-1 scale)"""
        # In real implementation, this would be actual user ratings
        # Simulate improving feedback over time
        base_score = random.uniform(0.4, 0.8)

        # Add some improvement trend
        cycle = len(clrs.all_feedback) // clrs.cycle_size
        improvement = min(cycle * 0.05, 0.2)

        return min(base_score + improvement, 1.0)

    # Run simulation
    print("\nRunning CLRS simulation...")
    print("NOTE: This demo uses simulated generation and feedback.")
    print("For real experiments, provide actual model and user feedback.\n")

    try:
        summary = clrs.simulate_feedback_loop(
            generate_fn=generate_output,
            input_samples=input_samples,
            feedback_fn=generate_feedback,
            num_cycles=5
        )

        # Log results
        logger.log_summary(summary)

        # Visualize
        visualizer = MetaLearningVisualizer()

        drift_scores = [c.drift_score for c in clrs.training_cycles]
        alignment_scores = [c.alignment_score for c in clrs.training_cycles]
        coherence_scores = [c.coherence_score for c in clrs.training_cycles]

        visualizer.plot_clrs_metrics(
            drift_scores=drift_scores,
            alignment_scores=alignment_scores,
            coherence_scores=coherence_scores,
            title="CLRS Simulation Results",
            save_name="clrs_demo.png"
        )

        # Print analysis
        print("\n" + "="*70)
        print("DETAILED ANALYSIS")
        print("="*70)

        print(f"\nDrift Analysis:")
        print(f"  Current: {summary['drift']['current']:.4f}")
        print(f"  Average: {summary['drift']['average']:.4f}")
        print(f"  Trend: {summary['drift']['trend_direction']}")

        print(f"\nAlignment Analysis:")
        print(f"  Current: {summary['alignment']['current']:.4f}")
        print(f"  Average: {summary['alignment']['average']:.4f}")
        print(f"  Trend: {summary['alignment']['trend_direction']}")

        print(f"\nCoherence Analysis:")
        print(f"  Current: {summary['coherence']['current']:.4f}")
        print(f"  Average: {summary['coherence']['average']:.4f}")
        print(f"  Trend: {summary['coherence']['trend_direction']}")

        print(f"\nSystem Health: {summary['health_status'].upper()}")

        # Export results
        clrs.export_results()

        logger.close()

    except Exception as e:
        logger.log_error(e, context="clrs_simulation")
        raise


if __name__ == "__main__":
    main()
