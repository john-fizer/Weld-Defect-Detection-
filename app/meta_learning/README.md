# Meta-Learning & Self-Optimizing Systems

Advanced AI engineering framework for researching meta-learning, self-optimizing prompts, and closed-loop reinforcement systems.

## Overview

This module implements two complementary approaches to adaptive AI systems:

### A. Adaptive Curriculum Learning Agent (ACLA)
An LLM that rewrites its own training prompts to improve task accuracy across iterations.

**Key Features:**
- Self-optimizing prompt evolution
- Multiple evolution strategies (performance-based, error analysis, chain-of-thought, etc.)
- Performance tracking and convergence detection
- Automatic prompt improvement based on feedback

**Research Question:** Can meta-prompting outperform static fine-tuning?

### B. Closed-Loop Reinforcement System (CLRS)
A feedback engine where user input continuously trains and refines a local model.

**Key Features:**
- Feedback-driven model adaptation
- Drift detection and monitoring
- Alignment scoring with user preferences
- Coherence analysis for emergent patterns
- Multi-cycle reinforcement learning

**Research Questions:**
- How does model behavior drift over training cycles?
- Can alignment be maintained with minimal feedback?
- What emergent patterns appear in model coherence?

## Architecture

```
app/meta_learning/
├── acla/                          # Adaptive Curriculum Learning Agent
│   ├── curriculum_agent.py        # Main ACLA implementation
│   ├── prompt_evolver.py          # Prompt evolution engine
│   └── performance_tracker.py     # Performance monitoring
│
├── clrs/                          # Closed-Loop Reinforcement System
│   ├── reinforcement_system.py    # Main CLRS implementation
│   ├── drift_detector.py          # Drift detection & monitoring
│   ├── alignment_scorer.py        # Alignment scoring
│   └── coherence_analyzer.py      # Coherence analysis
│
├── datasets/                      # Dataset loaders
│   ├── base.py                    # Base dataset loader
│   ├── commonsense_qa.py          # CommonsenseQA dataset
│   └── sentiment140.py            # Sentiment140 dataset
│
├── experiments/                   # Experiment framework
│   ├── runner.py                  # Experiment orchestration
│   ├── evaluator.py               # Model evaluation
│   └── comparator.py              # Result comparison
│
└── utils/                         # Utilities
    ├── visualizer.py              # Visualization tools
    └── logger.py                  # Experiment logging
```

## Installation

```bash
# The meta-learning module is part of the main project
# Ensure all dependencies are installed:
pip install -r requirements.txt

# Additional optional dependencies for visualization:
pip install matplotlib seaborn
```

## Quick Start

### 1. Adaptive Curriculum Learning Agent (ACLA)

```python
import asyncio
from app.meta_learning.acla import AdaptiveCurriculumAgent, CurriculumConfig
from app.meta_learning.datasets import CommonsenseQALoader

# Load dataset
dataset = CommonsenseQALoader()
dataset.load_data()

# Configure ACLA
config = CurriculumConfig(
    initial_prompt="Answer the following question...",
    dataset_name="commonsense_qa",
    max_iterations=10,
    min_performance_threshold=0.8
)

# Initialize agent
agent = AdaptiveCurriculumAgent(
    config=config,
    llm_client=your_llm_client  # Anthropic or OpenAI client
)

# Run curriculum learning
async def evaluate_fn(prompt, samples):
    # Your evaluation logic
    pass

results = await agent.run_curriculum(
    dataset_loader=lambda n: dataset.get_samples(n),
    evaluation_fn=evaluate_fn,
    num_samples=100
)

print(f"Best performance: {results['best_performance']}")
print(f"Best prompt: {results['best_prompt']}")
```

### 2. Closed-Loop Reinforcement System (CLRS)

```python
from app.meta_learning.clrs import ClosedLoopSystem

# Initialize CLRS
clrs = ClosedLoopSystem(
    cycle_size=100,           # Samples per training cycle
    drift_threshold=0.3,      # Max acceptable drift
    alignment_threshold=0.7   # Min alignment score
)

# Collect feedback
clrs.collect_feedback(
    input_text="user input",
    output_text="model output",
    feedback_score=0.8,  # 0-1 scale
    metadata={"source": "user"}
)

# System automatically trains when cycle_size is reached

# Get system summary
summary = clrs.get_system_summary()
print(f"Drift: {summary['drift']['current']}")
print(f"Alignment: {summary['alignment']['current']}")
print(f"Coherence: {summary['coherence']['current']}")
```

### 3. Run Comparison Study

```python
from app.meta_learning.experiments import ExperimentRunner
from app.meta_learning.datasets import Sentiment140Loader

# Initialize
runner = ExperimentRunner(llm_client=your_client)
dataset = Sentiment140Loader()

# Run comparison: meta-prompting vs static baseline
comparison = await runner.run_comparison_study(
    dataset_name="sentiment140",
    dataset_loader=lambda n: dataset.get_samples(n),
    initial_prompt="Classify the sentiment...",
    num_iterations=10,
    sample_size=100
)

# Results
if comparison['meta_prompting_wins']:
    print("✓ Meta-prompting outperforms static baseline!")
    print(f"Win margin: {comparison['win_margin']:.4f}")
```

## Examples

See `examples/meta_learning/` for complete working examples:

- `example_acla.py` - Adaptive Curriculum Learning Agent demo
- `example_clrs.py` - Closed-Loop Reinforcement System demo
- `example_comparison.py` - Meta-prompting vs baseline comparison

Run examples:
```bash
# ACLA example
python examples/meta_learning/example_acla.py

# CLRS example
python examples/meta_learning/example_clrs.py

# Comparison study
python examples/meta_learning/example_comparison.py
```

## Research Applications

### 1. Meta-Prompting Research

**Question:** Can meta-prompting outperform static fine-tuning?

**Approach:**
- Test ACLA on multiple datasets (CommonsenseQA, Sentiment140, etc.)
- Compare against static baseline
- Measure convergence speed, final performance, and robustness

**Key Metrics:**
- Final accuracy improvement
- Iterations to convergence
- Strategy effectiveness
- Generalization to new tasks

### 2. Feedback Loop Dynamics

**Questions:**
- How does model behavior drift over training cycles?
- Can alignment be maintained with minimal feedback?
- What emergent coherence patterns appear?

**Approach:**
- Deploy CLRS with real user feedback
- Monitor drift, alignment, coherence over time
- Analyze emergent patterns and behaviors

**Key Metrics:**
- Drift score trends
- Alignment stability
- Coherence patterns
- User satisfaction correlation

### 3. Evolution Strategy Analysis

**Question:** Which prompt evolution strategies work best?

**Approach:**
- Test different evolution strategies
- Compare effectiveness across datasets
- Identify optimal strategy combinations

**Strategies:**
- `performance_based` - Optimize based on metrics
- `error_analysis` - Focus on fixing errors
- `ablation` - Remove/add components systematically
- `chain_of_thought` - Add reasoning scaffolding
- `few_shot_optimization` - Improve examples

## Datasets

### CommonsenseQA
- **Task:** Multiple-choice commonsense reasoning
- **Size:** Sample dataset included (expandable)
- **Metrics:** Accuracy, precision, recall, F1

### Sentiment140
- **Task:** Binary sentiment classification (positive/negative)
- **Size:** Sample dataset included (expandable)
- **Metrics:** Accuracy, precision, recall, F1

### Custom Datasets
Extend `BaseDatasetLoader` to add your own datasets:

```python
from app.meta_learning.datasets import BaseDatasetLoader

class MyDatasetLoader(BaseDatasetLoader):
    def load_data(self):
        # Load your data
        pass

    def format_sample(self, sample):
        # Format for evaluation
        return {
            'input': sample['text'],
            'expected': sample['label'],
            'metadata': {}
        }
```

## Visualization

The framework includes comprehensive visualization tools:

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

# Create comprehensive dashboard
visualizer.create_dashboard(
    experiment_results=results,
    save_name="dashboard.png"
)
```

## Logging

Structured logging for experiments:

```python
from app.meta_learning.utils import ExperimentLogger

logger = ExperimentLogger(experiment_name="my_experiment")

logger.log_config({'dataset': 'commonsense_qa', 'iterations': 10})
logger.log_iteration(iteration=1, metrics={'accuracy': 0.75})
logger.log_event('convergence', 'Model converged at iteration 5')
logger.log_summary({'final_accuracy': 0.85, 'improvement': 0.10})
logger.close()
```

Logs are saved as both text and JSON for analysis.

## Configuration

### ACLA Configuration

```python
CurriculumConfig(
    initial_prompt: str,              # Starting prompt
    dataset_name: str,                # Dataset identifier
    max_iterations: int = 10,         # Max learning iterations
    min_performance_threshold: float = 0.7,  # Target performance
    improvement_threshold: float = 0.05,     # Min improvement to continue
    evolution_strategies: List[str] = [...], # Evolution strategies
    llm_provider: str = "anthropic",  # "anthropic" or "openai"
    model_name: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.7
)
```

### CLRS Configuration

```python
ClosedLoopSystem(
    model: Optional[Any] = None,      # Model to train
    cycle_size: int = 100,            # Samples per cycle
    drift_threshold: float = 0.3,     # Max acceptable drift
    alignment_threshold: float = 0.7  # Min alignment score
)
```

## Performance Considerations

- **ACLA**: Each iteration requires LLM calls for both evaluation and prompt evolution. Plan accordingly for API costs and latency.
- **CLRS**: Training cycles are triggered automatically. Adjust `cycle_size` based on your feedback volume.
- **Datasets**: Start with small sample sizes for testing, then scale up for production experiments.

## Research Output

All experiments automatically save:
- **Results:** JSON files with complete metrics
- **Checkpoints:** Model/prompt states at each iteration
- **Logs:** Structured text and JSON logs
- **Visualizations:** Charts and dashboards

Default save location: `./data/meta_learning/`

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{meta_learning_framework,
  title = {Meta-Learning \& Self-Optimizing Systems},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/repository}
}
```

## Contributing

Contributions welcome! Areas of interest:
- Additional evolution strategies
- New dataset loaders
- Alternative drift/alignment metrics
- Improved visualization tools
- Performance optimizations

## License

MIT License - see LICENSE file for details

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the examples directory
- Review the inline documentation

## Roadmap

- [ ] Support for more LLM providers (Cohere, local models)
- [ ] Ensemble meta-prompting (combine multiple evolved prompts)
- [ ] Multi-objective optimization (accuracy + efficiency)
- [ ] Automated hyperparameter tuning
- [ ] Real-time dashboard for live experiments
- [ ] Integration with MLflow/Weights & Biases
- [ ] Distributed training support
- [ ] Model compression for CLRS

## Acknowledgments

Built on top of:
- Anthropic Claude API
- OpenAI API
- scikit-learn for metrics
- matplotlib for visualization
- PyTorch for ML foundations
