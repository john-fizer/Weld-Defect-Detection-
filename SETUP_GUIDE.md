# Meta-Learning Framework - Setup Guide

Complete setup and usage guide for the Meta-Learning & Self-Optimizing Systems framework.

## Quick Start (5 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run validation to ensure everything works
python scripts/validate_meta_learning.py

# 3. Run quick start demo
python scripts/quick_start.py

# 4. Try an example
python examples/meta_learning/example_clrs.py
```

## Detailed Installation

### Prerequisites

- Python 3.11+
- pip
- Virtual environment (recommended)

### Step 1: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**Key dependencies:**
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning metrics
- `matplotlib` - Visualization
- `anthropic` - Claude API (optional, for real LLM experiments)
- `openai` - OpenAI API (optional, for real LLM experiments)
- `pydantic` - Data validation
- `pydantic-settings` - Configuration management

### Step 3: Verify Installation

```bash
# Run validation script
python scripts/validate_meta_learning.py
```

Expected output:
```
✓ Imports Validated: 8
✓ Configuration Items: 6
✓ Functionality Tests: 9
✓ No Bugs Found!
STATUS: VALIDATION PASSED - All tests successful!
```

### Step 4: Configure (Optional)

Create a `.env` file in the project root:

```bash
# LLM API Keys (optional - for real experiments)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Paths (optional - defaults are fine)
META_LEARNING_DATA_PATH=./data/meta_learning
META_LEARNING_DATASETS_PATH=./data/datasets

# Debug mode
DEBUG=True
LOG_LEVEL=INFO
```

## Framework Structure

```
app/meta_learning/
├── acla/                  # Adaptive Curriculum Learning Agent
│   ├── curriculum_agent.py
│   ├── prompt_evolver.py
│   └── performance_tracker.py
│
├── clrs/                  # Closed-Loop Reinforcement System
│   ├── reinforcement_system.py
│   ├── drift_detector.py
│   ├── alignment_scorer.py
│   └── coherence_analyzer.py
│
├── datasets/              # Dataset loaders
│   ├── commonsense_qa.py
│   └── sentiment140.py
│
├── experiments/           # Experiment framework
│   ├── runner.py
│   ├── evaluator.py
│   └── comparator.py
│
└── utils/                 # Utilities
    ├── visualizer.py
    └── logger.py
```

## Usage Examples

### Example 1: Basic Dataset Loading

```python
from app.meta_learning.datasets import CommonsenseQALoader

# Load dataset
dataset = CommonsenseQALoader()
dataset.load_data()

# Get samples
samples = dataset.get_samples(num_samples=10, split='test')

# Show statistics
stats = dataset.get_statistics()
print(f"Total samples: {stats['total_samples']}")
```

### Example 2: Drift Detection

```python
from app.meta_learning.clrs import DriftDetector

# Create detector
detector = DriftDetector()

# Set baseline
baseline = ["response 1", "response 2", "response 3"]
detector.set_baseline(baseline)

# Check for drift
current = ["new response 1", "new response 2", "new response 3"]
drift_score = detector.calculate_drift(current)

print(f"Drift score: {drift_score:.4f}")  # 0 = no drift, 1 = max drift
```

### Example 3: CLRS with Feedback

```python
from app.meta_learning.clrs import ClosedLoopSystem

# Initialize system
clrs = ClosedLoopSystem(cycle_size=100)

# Collect feedback
clrs.collect_feedback(
    input_text="User question",
    output_text="Model response",
    feedback_score=0.8,  # 0-1 scale
    metadata={'source': 'user'}
)

# Training happens automatically when cycle_size is reached

# Get system health
summary = clrs.get_system_summary()
print(f"Drift: {summary['drift']['current']:.4f}")
print(f"Alignment: {summary['alignment']['current']:.4f}")
print(f"Health: {summary['health_status']}")
```

### Example 4: Adaptive Curriculum Learning

```python
import asyncio
from app.meta_learning.acla import AdaptiveCurriculumAgent, CurriculumConfig
from app.meta_learning.datasets import Sentiment140Loader

async def run_acla():
    # Load dataset
    dataset = Sentiment140Loader()
    dataset.load_data()

    # Configure ACLA
    config = CurriculumConfig(
        initial_prompt="Classify the sentiment...",
        dataset_name="sentiment140",
        max_iterations=10
    )

    # Create agent
    agent = AdaptiveCurriculumAgent(config=config)

    # Define evaluation function
    async def evaluate(prompt, samples):
        # Your evaluation logic here
        return {'accuracy': 0.75}

    # Run curriculum learning
    results = await agent.run_curriculum(
        dataset_loader=lambda n: dataset.get_samples(n),
        evaluation_fn=evaluate,
        num_samples=100
    )

    print(f"Best prompt: {results['best_prompt']}")
    print(f"Best performance: {results['best_performance']}")

# Run
asyncio.run(run_acla())
```

### Example 5: Full Comparison Study

```python
import asyncio
from app.meta_learning.experiments import ExperimentRunner
from app.meta_learning.datasets import CommonsenseQALoader

async def run_comparison():
    # Initialize
    runner = ExperimentRunner(llm_client=your_llm_client)
    dataset = CommonsenseQALoader()

    # Run comparison: meta-prompting vs static baseline
    comparison = await runner.run_comparison_study(
        dataset_name="commonsense_qa",
        dataset_loader=lambda n: dataset.get_samples(n),
        initial_prompt="Answer the question...",
        num_iterations=10,
        sample_size=50
    )

    # Results
    if comparison['meta_prompting_wins']:
        print("✓ Meta-prompting outperforms static baseline!")
        print(f"Win margin: {comparison['win_margin']:.4f}")

    return comparison

# Run
asyncio.run(run_comparison())
```

## Running Examples

### 1. Quick Start Demo

```bash
python scripts/quick_start.py
```

Demonstrates all core features with interactive demos.

### 2. Validation

```bash
python scripts/validate_meta_learning.py
```

Runs comprehensive tests on all components.

### 3. ACLA Example

```bash
python examples/meta_learning/example_acla.py
```

Shows adaptive curriculum learning on CommonsenseQA.

### 4. CLRS Example

```bash
python examples/meta_learning/example_clrs.py
```

Demonstrates closed-loop reinforcement with feedback.

### 5. Comparison Study

```bash
python examples/meta_learning/example_comparison.py
```

Compares meta-prompting vs static baseline on Sentiment140.

## Connecting Real LLMs

To use real LLMs instead of simulated evaluation:

### Using Anthropic Claude

```python
import anthropic
from app.meta_learning.acla import AdaptiveCurriculumAgent, CurriculumConfig

# Create client
client = anthropic.Anthropic(api_key="your-key-here")

# Create agent with client
config = CurriculumConfig(
    initial_prompt="...",
    dataset_name="commonsense_qa",
    llm_provider="anthropic",
    model_name="claude-3-5-sonnet-20241022"
)

agent = AdaptiveCurriculumAgent(
    config=config,
    llm_client=client  # Pass the client
)

# Now the agent will use real Claude API for prompt evolution
```

### Using OpenAI

```python
import openai
from app.meta_learning.acla import CurriculumConfig, AdaptiveCurriculumAgent

# Create client
client = openai.OpenAI(api_key="your-key-here")

# Create agent
config = CurriculumConfig(
    initial_prompt="...",
    dataset_name="sentiment140",
    llm_provider="openai",
    model_name="gpt-4"
)

agent = AdaptiveCurriculumAgent(
    config=config,
    llm_client=client
)
```

## Visualization

Generate visualizations for your experiments:

```python
from app.meta_learning.utils import MetaLearningVisualizer

visualizer = MetaLearningVisualizer()

# Plot curriculum learning progress
visualizer.plot_curriculum_learning(
    performance_history=results['performance_history'],
    save_name="acla_progress.png"
)

# Plot comparison
visualizer.plot_comparison(
    meta_prompting_history=meta_history,
    baseline_history=baseline_history,
    save_name="comparison.png"
)

# Plot CLRS metrics
visualizer.plot_clrs_metrics(
    drift_scores=drift_scores,
    alignment_scores=alignment_scores,
    coherence_scores=coherence_scores,
    save_name="clrs_metrics.png"
)
```

## Logging

Track your experiments with structured logging:

```python
from app.meta_learning.utils import ExperimentLogger

logger = ExperimentLogger(experiment_name="my_experiment")

# Log configuration
logger.log_config({
    'dataset': 'commonsense_qa',
    'iterations': 10
})

# Log iterations
logger.log_iteration(1, {'accuracy': 0.75, 'loss': 0.25})

# Log events
logger.log_event('convergence', 'Converged at iteration 5')

# Close logger
logger.close()
```

Logs are saved in both text (.log) and JSON (.jsonl) formats.

## Troubleshooting

### Import Errors

If you get import errors:
```bash
# Make sure you're in the project root directory
cd /path/to/Weld-Defect-Detection-

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Missing Dependencies

```bash
# Install specific package
pip install numpy pandas scikit-learn matplotlib

# Or install all
pip install -r requirements.txt
```

### Configuration Issues

```bash
# Run validation to identify issues
python scripts/validate_meta_learning.py

# Check Python version (need 3.11+)
python --version
```

### Dataset Download Failures

The framework creates sample datasets automatically if downloads fail.
This is normal and the samples are sufficient for testing.

### Visualization Issues

If matplotlib gives errors:
```bash
# Install with display backend
pip install matplotlib pillow

# Or use headless mode
import matplotlib
matplotlib.use('Agg')  # Before importing pyplot
```

## Next Steps

1. **Read the docs**: `app/meta_learning/README.md`
2. **Try examples**: Start with `example_clrs.py`
3. **Run experiments**: Use your own datasets
4. **Customize**: Extend the framework for your needs
5. **Contribute**: Share improvements back

## Getting Help

- Check examples in `examples/meta_learning/`
- Read inline documentation in source files
- Run validation script for diagnostics
- Review error messages from `validate_meta_learning.py`

## Performance Tips

1. **Use caching**: Datasets automatically cache for faster loading
2. **Batch operations**: Process multiple samples at once
3. **Adjust cycle_size**: Larger cycles = less frequent training
4. **Use simpler models**: For testing, use lightweight models
5. **Monitor memory**: Large datasets can consume RAM

## Best Practices

1. **Always validate**: Run `validate_meta_learning.py` after changes
2. **Use logging**: Track all experiments for reproducibility
3. **Version control**: Commit configurations and results
4. **Document experiments**: Use logger.log_config() religiously
5. **Test incrementally**: Start small, scale up
6. **Monitor health**: Check drift/alignment/coherence regularly

## Research Applications

This framework enables research into:

- **Meta-prompting effectiveness** vs traditional fine-tuning
- **Prompt evolution dynamics** and convergence patterns
- **Feedback loop behavior** in adaptive systems
- **Drift patterns** in continuously learning models
- **Alignment stability** with minimal feedback
- **Emergent coherence** in self-optimizing systems

Perfect for:
- Academic research papers
- NeurIPS/ICML/ICLR submissions
- Industrial ML/AI projects
- PhD dissertations
- Tech blog posts

## License

MIT License - see LICENSE file

## Support

For issues or questions:
1. Run `python scripts/validate_meta_learning.py`
2. Check error output
3. Review examples
4. Check documentation

Happy researching! 🚀
