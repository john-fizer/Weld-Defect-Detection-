# Meta-Learning & Self-Optimizing Systems

A comprehensive framework for advanced AI research into meta-learning, adaptive curriculum learning, and closed-loop reinforcement systems.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

This framework implements two complementary approaches to adaptive AI systems:

### A. Adaptive Curriculum Learning Agent (ACLA)
An LLM that **rewrites its own training prompts** to improve task accuracy across iterations.

**Research Question:** *Can meta-prompting outperform static fine-tuning?*

**Key Features:**
- 🔄 Self-optimizing prompt evolution
- 📊 Multiple evolution strategies (performance-based, error analysis, chain-of-thought, etc.)
- 📈 Performance tracking and convergence detection
- 🎯 Automatic prompt improvement based on feedback

### B. Closed-Loop Reinforcement System (CLRS)
A feedback engine where **user input continuously trains and refines a local model**.

**Research Questions:**
- How does model behavior drift over training cycles?
- Can alignment be maintained with minimal feedback?
- What emergent patterns appear in model coherence?

**Key Features:**
- 🔁 Feedback-driven model adaptation
- 📉 Drift detection and monitoring (vocabulary, distribution, length)
- 🎯 Alignment scoring with preference learning
- 🧩 Coherence analysis for emergent patterns
- 🏥 Multi-cycle reinforcement with health monitoring

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/meta-learning-framework.git
cd meta-learning-framework

# Install dependencies
pip install -r requirements.txt

# Validate installation
python scripts/validate_meta_learning.py

# Run quick start demo
python scripts/quick_start.py
```

### 5-Minute Demo

```bash
# Try the CLRS example (Closed-Loop Reinforcement System)
python examples/meta_learning/example_clrs.py
```

Output shows:
- Feedback collection and training cycles
- Drift, alignment, and coherence metrics
- System health monitoring
- Visualization generation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Meta-Learning Framework                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  ACLA            │         │  CLRS            │         │
│  │  ┌────────────┐  │         │  ┌────────────┐  │         │
│  │  │ Curriculum │  │         │  │ Feedback   │  │         │
│  │  │ Agent      │  │         │  │ Collection │  │         │
│  │  └────────────┘  │         │  └────────────┘  │         │
│  │  ┌────────────┐  │         │  ┌────────────┐  │         │
│  │  │ Prompt     │  │         │  │ Drift      │  │         │
│  │  │ Evolver    │  │         │  │ Detector   │  │         │
│  │  └────────────┘  │         │  └────────────┘  │         │
│  │  ┌────────────┐  │         │  ┌────────────┐  │         │
│  │  │ Performance│  │         │  │ Alignment  │  │         │
│  │  │ Tracker    │  │         │  │ Scorer     │  │         │
│  │  └────────────┘  │         │  └────────────┘  │         │
│  └──────────────────┘         └──────────────────┘         │
│                                                              │
│  ┌──────────────────────────────────────────────┐           │
│  │  Datasets (CommonsenseQA, Sentiment140)      │           │
│  └──────────────────────────────────────────────┘           │
│                                                              │
│  ┌──────────────────────────────────────────────┐           │
│  │  Experiments (Runner, Evaluator, Comparator) │           │
│  └──────────────────────────────────────────────┘           │
│                                                              │
│  ┌──────────────────────────────────────────────┐           │
│  │  Utils (Visualizer, Logger)                  │           │
│  └──────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## Features

### 🎓 Adaptive Curriculum Learning

```python
from app.meta_learning.acla import AdaptiveCurriculumAgent, CurriculumConfig

config = CurriculumConfig(
    initial_prompt="Answer the question...",
    dataset_name="commonsense_qa",
    max_iterations=10
)

agent = AdaptiveCurriculumAgent(config=config, llm_client=your_client)
results = await agent.run_curriculum(dataset_loader, evaluation_fn)

print(f"Best prompt: {results['best_prompt']}")
print(f"Improvement: {results['total_improvement']}")
```

### 🔄 Closed-Loop Reinforcement

```python
from app.meta_learning.clrs import ClosedLoopSystem

clrs = ClosedLoopSystem(cycle_size=100)

# Collect feedback
clrs.collect_feedback(
    input_text="User input",
    output_text="Model output",
    feedback_score=0.8
)

# Auto-trains when cycle completes
summary = clrs.get_system_summary()
print(f"Drift: {summary['drift']['current']}")
print(f"Alignment: {summary['alignment']['current']}")
```

### 📊 Experiment Comparison

```python
from app.meta_learning.experiments import ExperimentRunner

runner = ExperimentRunner(llm_client=your_client)

# Compare meta-prompting vs static baseline
comparison = await runner.run_comparison_study(
    dataset_name="sentiment140",
    dataset_loader=loader,
    initial_prompt="Classify sentiment...",
    num_iterations=10
)

if comparison['meta_prompting_wins']:
    print(f"✓ Meta-prompting wins by {comparison['win_margin']:.4f}")
```

## Research Applications

This framework enables research into:

- **Meta-prompting effectiveness** vs traditional fine-tuning
- **Prompt evolution dynamics** and convergence patterns
- **Feedback loop behavior** in adaptive systems
- **Drift patterns** in continuously learning models
- **Alignment stability** with minimal feedback
- **Emergent coherence** in self-optimizing systems

Perfect for:
- 📝 Academic research papers (NeurIPS, ICML, ICLR)
- 🏭 Industrial ML/AI projects
- 🎓 PhD dissertations
- 📰 Technical blog posts
- 🔬 AI safety and alignment research

## Components

### Core Framework

**ACLA (Adaptive Curriculum Learning Agent)**
- `curriculum_agent.py` - Main agent implementation
- `prompt_evolver.py` - Prompt evolution engine
- `performance_tracker.py` - Performance monitoring

**CLRS (Closed-Loop Reinforcement System)**
- `reinforcement_system.py` - Main system implementation
- `drift_detector.py` - Distribution drift detection
- `alignment_scorer.py` - Preference alignment scoring
- `coherence_analyzer.py` - Coherence pattern analysis

**Datasets**
- `commonsense_qa.py` - CommonsenseQA loader
- `sentiment140.py` - Sentiment140 loader
- `base.py` - Extensible base loader

**Experiments**
- `runner.py` - Experiment orchestration
- `evaluator.py` - Model evaluation
- `comparator.py` - Statistical comparison

**Utilities**
- `visualizer.py` - Comprehensive visualization
- `logger.py` - Structured experiment logging

## Examples

### Example 1: ACLA on CommonsenseQA

```bash
python examples/meta_learning/example_acla.py
```

Demonstrates:
- Adaptive curriculum learning
- Prompt evolution strategies
- Performance tracking
- Convergence detection

### Example 2: CLRS Simulation

```bash
python examples/meta_learning/example_clrs.py
```

Demonstrates:
- Feedback collection
- Drift detection
- Alignment scoring
- Coherence analysis
- Health monitoring

### Example 3: Meta-Prompting vs Baseline

```bash
python examples/meta_learning/example_comparison.py
```

Demonstrates:
- Full comparison study
- Statistical significance testing
- Visualization generation
- Result analysis

## Testing

```bash
# Run comprehensive validation
python scripts/validate_meta_learning.py
```

Tests:
- ✅ All imports (8 components)
- ✅ Configuration (6 paths)
- ✅ Functionality (9 components)
- ✅ Bug detection

Expected: `16/16 tests PASSED`

## Documentation

- **[Setup Guide](SETUP_GUIDE.md)** - Installation and configuration
- **[Implementation Summary](META_LEARNING_SUMMARY.md)** - Architecture details
- **[Fixes Summary](FIXES_SUMMARY.md)** - Development history
- **[Examples](examples/meta_learning/)** - Working code examples

## Requirements

- Python 3.11+
- NumPy, Pandas, scikit-learn
- Matplotlib (for visualization)
- Optional: PyTorch, Anthropic/OpenAI API (for real LLM experiments)

See `requirements.txt` for full list.

## Development

### Project Structure

```
meta-learning-framework/
├── app/meta_learning/      # Core framework
├── examples/               # Usage examples
├── scripts/               # Utility scripts
├── docs/                  # Documentation
└── tests/                 # Test suite
```

### Contributing

Contributions welcome! Areas of interest:
- Additional evolution strategies
- New dataset loaders
- Alternative drift/alignment metrics
- Performance optimizations
- More visualization options

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{meta_learning_framework_2024,
  title = {Meta-Learning and Self-Optimizing Systems Framework},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/YOUR_USERNAME/meta-learning-framework}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

Built using:
- Anthropic Claude API
- OpenAI API
- scikit-learn
- Matplotlib
- PyTorch

Developed with AI-assisted pair programming using Claude Code.

## Contact

- GitHub Issues: [Report bugs or request features]
- Email: your.email@example.com
- Website: your-website.com

---

**Ready to explore meta-learning?** Start with the [Quick Start](#quick-start) guide!
