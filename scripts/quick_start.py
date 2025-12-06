#!/usr/bin/env python
"""
Quick Start Script for Meta-Learning Framework

Run this script to:
1. Validate installation
2. Create sample directories
3. Run a quick demo
4. Generate example outputs
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings
from app.meta_learning.datasets import CommonsenseQALoader, Sentiment140Loader
from app.meta_learning.clrs import ClosedLoopSystem, DriftDetector
from app.meta_learning.acla import PerformanceTracker
from app.meta_learning.utils import MetaLearningVisualizer, ExperimentLogger
import random


def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*70}")
    print(f"{text:^70}")
    print(f"{'='*70}\n")


def setup_directories():
    """Create necessary directories"""
    print_header("Setting Up Directories")

    dirs = [
        Path(settings.meta_learning_data_path),
        Path(settings.meta_learning_datasets_path),
        Path(settings.meta_learning_experiments_path),
        Path(settings.meta_learning_checkpoints_path),
        Path(settings.meta_learning_logs_path),
        Path(settings.meta_learning_visualizations_path),
    ]

    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")

    print("\nAll directories ready!")


def demo_datasets():
    """Demonstrate dataset loading"""
    print_header("Dataset Loading Demo")

    print("1. Loading CommonsenseQA...")
    csqa = CommonsenseQALoader()
    csqa.load_data()
    stats = csqa.get_statistics()
    print(f"   ✓ Loaded {stats['total_samples']} samples")
    print(f"   ✓ Train: {stats['train_samples']}, Test: {stats['test_samples']}")

    # Show a sample
    sample = csqa.get_samples(num_samples=1, split='test')[0]
    print(f"\n   Sample question: {sample['question'][:60]}...")

    print("\n2. Loading Sentiment140...")
    s140 = Sentiment140Loader()
    s140.load_data()
    stats = s140.get_statistics()
    print(f"   ✓ Loaded {stats['total_samples']} samples")
    print(f"   ✓ Distribution: {stats['sentiment_distribution']}")

    # Show a sample
    sample = s140.get_samples(num_samples=1, split='test')[0]
    print(f"\n   Sample tweet: {sample['text'][:60]}...")


def demo_drift_detection():
    """Demonstrate drift detection"""
    print_header("Drift Detection Demo")

    detector = DriftDetector()

    print("1. Setting baseline...")
    baseline = [
        "This is a helpful response about machine learning.",
        "Let me explain the concept clearly.",
        "Here's a detailed answer to your question."
    ]
    detector.set_baseline(baseline)
    print("   ✓ Baseline established")

    print("\n2. Testing drift with similar outputs...")
    similar = [
        "This is a helpful answer about AI.",
        "Let me clarify the concept.",
        "Here's a comprehensive response."
    ]
    drift_similar = detector.calculate_drift(similar)
    print(f"   ✓ Drift score (similar): {drift_similar:.4f}")

    print("\n3. Testing drift with different outputs...")
    different = [
        "Random text here.",
        "Completely unrelated content.",
        "Something totally different."
    ]
    drift_different = detector.calculate_drift(different)
    print(f"   ✓ Drift score (different): {drift_different:.4f}")

    print(f"\n   Analysis: Higher drift with different content ({drift_different:.4f} vs {drift_similar:.4f})")


def demo_clrs():
    """Demonstrate CLRS"""
    print_header("Closed-Loop Reinforcement System Demo")

    clrs = ClosedLoopSystem(cycle_size=5)

    print("1. Simulating feedback collection...")

    # Simulate 10 feedback samples
    inputs = [
        "What is AI?",
        "Explain machine learning",
        "How does neural network work?",
        "What is deep learning?",
        "Describe reinforcement learning"
    ] * 2

    for inp in inputs:
        output = f"Here's an explanation of {inp[:20]}..."
        feedback = random.uniform(0.6, 0.9)  # Simulate user feedback

        clrs.collect_feedback(
            input_text=inp,
            output_text=output,
            feedback_score=feedback
        )

    print(f"   ✓ Collected {len(clrs.all_feedback)} feedback samples")
    print(f"   ✓ Completed {len(clrs.training_cycles)} training cycles")

    if clrs.training_cycles:
        last_cycle = clrs.training_cycles[-1]
        print(f"\n2. Last cycle metrics:")
        print(f"   - Drift: {last_cycle.drift_score:.4f}")
        print(f"   - Alignment: {last_cycle.alignment_score:.4f}")
        print(f"   - Coherence: {last_cycle.coherence_score:.4f}")


def demo_performance_tracking():
    """Demonstrate performance tracking"""
    print_header("Performance Tracking Demo")

    tracker = PerformanceTracker()

    print("1. Recording iterations...")

    # Simulate improving performance
    for i in range(5):
        accuracy = 0.5 + (i * 0.08)  # Improving from 0.5 to 0.82
        f1 = 0.45 + (i * 0.08)

        tracker.record_iteration(i, {
            'accuracy': accuracy,
            'f1': f1
        })

    print(f"   ✓ Recorded {len(tracker.history)} iterations")

    summary = tracker.get_summary_statistics()

    print("\n2. Performance summary:")
    if 'accuracy' in summary['metrics']:
        acc_stats = summary['metrics']['accuracy']
        print(f"   - Initial accuracy: {acc_stats['initial']:.4f}")
        print(f"   - Final accuracy: {acc_stats['final']:.4f}")
        print(f"   - Improvement: {acc_stats['improvement']:.4f}")
        if 'convergence' in acc_stats and 'is_improving' in acc_stats['convergence']:
            trend = "improving" if acc_stats['convergence']['is_improving'] else "stable"
            print(f"   - Trend: {trend}")


def demo_logging():
    """Demonstrate logging"""
    print_header("Experiment Logging Demo")

    import tempfile
    import shutil

    temp_dir = Path(tempfile.mkdtemp())

    try:
        logger = ExperimentLogger(
            log_dir=temp_dir,
            experiment_name="quick_start_demo"
        )

        print("1. Logging experiment configuration...")
        logger.log_config({
            'dataset': 'CommonsenseQA',
            'max_iterations': 5,
            'model': 'claude-3-5-sonnet'
        })

        print("2. Logging iterations...")
        for i in range(3):
            logger.log_iteration(i + 1, {
                'accuracy': 0.6 + i * 0.1,
                'loss': 0.4 - i * 0.1
            })

        print("3. Logging events...")
        logger.log_event('convergence', 'Model converged at iteration 3')

        logger.close()

        # Show log files
        log_files = list(temp_dir.glob("*"))
        print(f"\n   ✓ Created {len(log_files)} log files:")
        for log_file in log_files:
            print(f"      - {log_file.name}")

    finally:
        shutil.rmtree(temp_dir)
        print("   ✓ Cleaned up demo logs")


def run_all_demos():
    """Run all demos"""
    print_header("Meta-Learning Framework Quick Start")
    print("Welcome to the Meta-Learning & Self-Optimizing Systems framework!")
    print("\nThis script will demonstrate key features:")
    print("  1. Directory setup")
    print("  2. Dataset loading")
    print("  3. Drift detection")
    print("  4. Closed-loop reinforcement")
    print("  5. Performance tracking")
    print("  6. Experiment logging")

    input("\nPress Enter to continue...")

    try:
        setup_directories()
        demo_datasets()
        demo_drift_detection()
        demo_clrs()
        demo_performance_tracking()
        demo_logging()

        print_header("Quick Start Complete!")
        print("✓ All demos completed successfully!")
        print("\nNext steps:")
        print("  1. Check examples/meta_learning/ for detailed examples")
        print("  2. Run validation: python scripts/validate_meta_learning.py")
        print("  3. Try ACLA: python examples/meta_learning/example_acla.py")
        print("  4. Try CLRS: python examples/meta_learning/example_clrs.py")
        print("  5. Read docs: app/meta_learning/README.md")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_demos()
