"""
Experiment Framework for Meta-Learning Research

Compares:
- Meta-prompting vs static fine-tuning
- Different evolution strategies
- Feedback loop configurations
"""

from .runner import ExperimentRunner
from .evaluator import ModelEvaluator
from .comparator import ExperimentComparator

__all__ = [
    'ExperimentRunner',
    'ModelEvaluator',
    'ExperimentComparator',
]
