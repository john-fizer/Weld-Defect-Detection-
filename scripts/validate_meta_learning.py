"""
Validation Script for Meta-Learning Framework

Tests:
- Import validation
- Configuration validation
- Basic functionality tests
- Example code execution
- Bug detection
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*70)
print("META-LEARNING FRAMEWORK VALIDATION")
print("="*70)

# Track results
results = {
    'imports': [],
    'config': [],
    'functionality': [],
    'bugs': []
}

def test_section(name):
    """Decorator for test sections"""
    def decorator(func):
        def wrapper():
            print(f"\n{'='*70}")
            print(f"Testing: {name}")
            print(f"{'='*70}")
            try:
                func()
                print(f"✓ {name} - PASSED")
                return True
            except Exception as e:
                print(f"✗ {name} - FAILED")
                print(f"Error: {str(e)}")
                traceback.print_exc()
                results['bugs'].append({
                    'section': name,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                return False
        return wrapper
    return decorator


@test_section("Import Core Modules")
def test_core_imports():
    """Test importing core modules"""
    print("\n1. Testing app package...")
    import app
    print(f"   ✓ app version: {app.__version__}")
    results['imports'].append('app')

    print("\n2. Testing app.config...")
    from app.config import settings
    print(f"   ✓ config loaded")
    print(f"   - Debug mode: {settings.debug}")
    print(f"   - Meta-learning data path: {settings.meta_learning_data_path}")
    results['imports'].append('app.config')

    print("\n3. Testing app.meta_learning...")
    from app import meta_learning
    print(f"   ✓ meta_learning module loaded")
    results['imports'].append('app.meta_learning')


@test_section("Import ACLA Components")
def test_acla_imports():
    """Test ACLA imports"""
    print("\n1. Importing ACLA...")
    from app.meta_learning.acla import AdaptiveCurriculumAgent, CurriculumConfig
    print("   ✓ AdaptiveCurriculumAgent imported")
    print("   ✓ CurriculumConfig imported")
    results['imports'].append('ACLA')

    print("\n2. Importing PromptEvolver...")
    from app.meta_learning.acla import PromptEvolver
    print("   ✓ PromptEvolver imported")

    print("\n3. Importing PerformanceTracker...")
    from app.meta_learning.acla import PerformanceTracker
    print("   ✓ PerformanceTracker imported")


@test_section("Import CLRS Components")
def test_clrs_imports():
    """Test CLRS imports"""
    print("\n1. Importing CLRS...")
    from app.meta_learning.clrs import ClosedLoopSystem
    print("   ✓ ClosedLoopSystem imported")
    results['imports'].append('CLRS')

    print("\n2. Importing DriftDetector...")
    from app.meta_learning.clrs import DriftDetector
    print("   ✓ DriftDetector imported")

    print("\n3. Importing AlignmentScorer...")
    from app.meta_learning.clrs import AlignmentScorer
    print("   ✓ AlignmentScorer imported")

    print("\n4. Importing CoherenceAnalyzer...")
    from app.meta_learning.clrs import CoherenceAnalyzer
    print("   ✓ CoherenceAnalyzer imported")


@test_section("Import Dataset Components")
def test_dataset_imports():
    """Test dataset imports"""
    print("\n1. Importing BaseDatasetLoader...")
    from app.meta_learning.datasets import BaseDatasetLoader
    print("   ✓ BaseDatasetLoader imported")
    results['imports'].append('Datasets')

    print("\n2. Importing CommonsenseQALoader...")
    from app.meta_learning.datasets import CommonsenseQALoader
    print("   ✓ CommonsenseQALoader imported")

    print("\n3. Importing Sentiment140Loader...")
    from app.meta_learning.datasets import Sentiment140Loader
    print("   ✓ Sentiment140Loader imported")


@test_section("Import Experiment Components")
def test_experiment_imports():
    """Test experiment imports"""
    print("\n1. Importing ExperimentRunner...")
    from app.meta_learning.experiments import ExperimentRunner
    print("   ✓ ExperimentRunner imported")
    results['imports'].append('Experiments')

    print("\n2. Importing ModelEvaluator...")
    from app.meta_learning.experiments import ModelEvaluator
    print("   ✓ ModelEvaluator imported")

    print("\n3. Importing ExperimentComparator...")
    from app.meta_learning.experiments import ExperimentComparator
    print("   ✓ ExperimentComparator imported")


@test_section("Import Utils Components")
def test_utils_imports():
    """Test utils imports"""
    print("\n1. Importing MetaLearningVisualizer...")
    from app.meta_learning.utils import MetaLearningVisualizer
    print("   ✓ MetaLearningVisualizer imported")
    results['imports'].append('Utils')

    print("\n2. Importing ExperimentLogger...")
    from app.meta_learning.utils import ExperimentLogger
    print("   ✓ ExperimentLogger imported")


@test_section("Test Configuration")
def test_configuration():
    """Test configuration settings"""
    from app.config import settings

    print("\n1. Checking meta-learning paths...")
    paths_to_check = [
        'meta_learning_data_path',
        'meta_learning_datasets_path',
        'meta_learning_experiments_path',
        'meta_learning_checkpoints_path',
        'meta_learning_logs_path',
        'meta_learning_visualizations_path'
    ]

    for path_name in paths_to_check:
        path_value = getattr(settings, path_name)
        print(f"   ✓ {path_name}: {path_value}")
        results['config'].append(path_name)


@test_section("Test ACLA Instantiation")
def test_acla_instantiation():
    """Test creating ACLA objects"""
    from app.meta_learning.acla import CurriculumConfig, AdaptiveCurriculumAgent

    print("\n1. Creating CurriculumConfig...")
    config = CurriculumConfig(
        initial_prompt="Test prompt",
        dataset_name="test_dataset",
        max_iterations=3
    )
    print(f"   ✓ Config created with {config.max_iterations} max iterations")

    print("\n2. Creating AdaptiveCurriculumAgent...")
    agent = AdaptiveCurriculumAgent(config=config)
    print(f"   ✓ Agent created")
    print(f"   ✓ Current prompt set: {len(agent.current_prompt) > 0}")
    results['functionality'].append('ACLA instantiation')


@test_section("Test CLRS Instantiation")
def test_clrs_instantiation():
    """Test creating CLRS objects"""
    from app.meta_learning.clrs import ClosedLoopSystem

    print("\n1. Creating ClosedLoopSystem...")
    clrs = ClosedLoopSystem(cycle_size=10)
    print(f"   ✓ CLRS created with cycle_size={clrs.cycle_size}")

    print("\n2. Testing feedback collection...")
    clrs.collect_feedback(
        input_text="test input",
        output_text="test output",
        feedback_score=0.8
    )
    print(f"   ✓ Feedback collected: {len(clrs.feedback_buffer)} samples")
    results['functionality'].append('CLRS instantiation')


@test_section("Test Dataset Loading")
def test_dataset_loading():
    """Test dataset loaders"""
    from app.meta_learning.datasets import CommonsenseQALoader, Sentiment140Loader

    print("\n1. Testing CommonsenseQA...")
    csqa = CommonsenseQALoader()
    csqa.load_data()
    stats = csqa.get_statistics()
    print(f"   ✓ Loaded {stats['total_samples']} samples")

    print("\n2. Testing Sentiment140...")
    s140 = Sentiment140Loader()
    s140.load_data()
    stats = s140.get_statistics()
    print(f"   ✓ Loaded {stats['total_samples']} samples")

    results['functionality'].append('Dataset loading')


@test_section("Test Drift Detection")
def test_drift_detection():
    """Test drift detector"""
    from app.meta_learning.clrs import DriftDetector

    print("\n1. Creating DriftDetector...")
    detector = DriftDetector()

    print("\n2. Setting baseline...")
    baseline = ["This is a test.", "Another test.", "More test data."]
    detector.set_baseline(baseline)
    print(f"   ✓ Baseline set")

    print("\n3. Calculating drift...")
    current = ["This is different.", "Very different.", "Completely new."]
    drift_score = detector.calculate_drift(current)
    print(f"   ✓ Drift score: {drift_score:.4f}")

    results['functionality'].append('Drift detection')


@test_section("Test Alignment Scoring")
def test_alignment_scoring():
    """Test alignment scorer"""
    from app.meta_learning.clrs import AlignmentScorer

    print("\n1. Creating AlignmentScorer...")
    scorer = AlignmentScorer()

    print("\n2. Calculating alignment...")
    inputs = ["input1", "input2", "input3"]
    outputs = ["output1", "output2", "output3"]
    feedbacks = [0.8, 0.7, 0.9]

    alignment = scorer.calculate_alignment(inputs, outputs, feedbacks)
    print(f"   ✓ Alignment score: {alignment:.4f}")

    results['functionality'].append('Alignment scoring')


@test_section("Test Coherence Analysis")
def test_coherence_analysis():
    """Test coherence analyzer"""
    from app.meta_learning.clrs import CoherenceAnalyzer

    print("\n1. Creating CoherenceAnalyzer...")
    analyzer = CoherenceAnalyzer()

    print("\n2. Calculating coherence...")
    outputs = [
        "This is a coherent response.",
        "This is another coherent response.",
        "This is a third coherent response."
    ]

    coherence = analyzer.calculate_coherence(outputs)
    print(f"   ✓ Coherence score: {coherence:.4f}")

    results['functionality'].append('Coherence analysis')


@test_section("Test Performance Tracker")
def test_performance_tracker():
    """Test performance tracker"""
    from app.meta_learning.acla import PerformanceTracker

    print("\n1. Creating PerformanceTracker...")
    tracker = PerformanceTracker()

    print("\n2. Recording iterations...")
    tracker.record_iteration(0, {'accuracy': 0.6, 'f1': 0.55})
    tracker.record_iteration(1, {'accuracy': 0.7, 'f1': 0.68})
    tracker.record_iteration(2, {'accuracy': 0.75, 'f1': 0.72})
    print(f"   ✓ Recorded 3 iterations")

    print("\n3. Getting summary...")
    summary = tracker.get_summary_statistics()
    print(f"   ✓ Summary generated for {summary['total_iterations']} iterations")

    results['functionality'].append('Performance tracking')


@test_section("Test Visualizer")
def test_visualizer():
    """Test visualizer"""
    from app.meta_learning.utils import MetaLearningVisualizer

    print("\n1. Creating MetaLearningVisualizer...")
    visualizer = MetaLearningVisualizer()
    print(f"   ✓ Visualizer created")
    print(f"   ✓ Save path: {visualizer.save_path}")

    results['functionality'].append('Visualizer')


@test_section("Test Logger")
def test_logger():
    """Test logger"""
    from app.meta_learning.utils import ExperimentLogger
    import tempfile
    import shutil

    print("\n1. Creating ExperimentLogger...")
    temp_dir = Path(tempfile.mkdtemp())

    try:
        logger = ExperimentLogger(
            log_dir=temp_dir,
            experiment_name="test_experiment"
        )
        print(f"   ✓ Logger created")

        print("\n2. Testing logging methods...")
        logger.log_config({'test': 'config'})
        logger.log_iteration(1, {'accuracy': 0.8})
        logger.log_event('test_event', 'Test message')
        logger.close()
        print(f"   ✓ All logging methods work")

        results['functionality'].append('Logger')
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def print_summary():
    """Print validation summary"""
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    print(f"\n✓ Imports Validated: {len(results['imports'])}")
    for imp in results['imports']:
        print(f"  - {imp}")

    print(f"\n✓ Configuration Items: {len(results['config'])}")
    for cfg in results['config']:
        print(f"  - {cfg}")

    print(f"\n✓ Functionality Tests: {len(results['functionality'])}")
    for func in results['functionality']:
        print(f"  - {func}")

    if results['bugs']:
        print(f"\n✗ Bugs Found: {len(results['bugs'])}")
        for bug in results['bugs']:
            print(f"\n  Bug in: {bug['section']}")
            print(f"  Error: {bug['error']}")
    else:
        print(f"\n✓ No Bugs Found!")

    print("\n" + "="*70)

    if results['bugs']:
        print("STATUS: VALIDATION FAILED - Bugs need fixing")
        return False
    else:
        print("STATUS: VALIDATION PASSED - All tests successful!")
        return True


# Run all tests
if __name__ == "__main__":
    test_core_imports()
    test_acla_imports()
    test_clrs_imports()
    test_dataset_imports()
    test_experiment_imports()
    test_utils_imports()
    test_configuration()
    test_acla_instantiation()
    test_clrs_instantiation()
    test_dataset_loading()
    test_drift_detection()
    test_alignment_scoring()
    test_coherence_analysis()
    test_performance_tracker()
    test_visualizer()
    test_logger()

    # Print summary
    success = print_summary()

    sys.exit(0 if success else 1)
