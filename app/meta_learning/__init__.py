"""
Meta-Learning & Self-Optimizing Systems

Advanced AI engineering framework for:
- Adaptive Curriculum Learning Agents (ACLA)
- Closed-Loop Reinforcement Systems (CLRS)
- Meta-prompting research
- Self-optimizing model pipelines
"""

from .acla.curriculum_agent import AdaptiveCurriculumAgent
from .clrs.reinforcement_system import ClosedLoopSystem
from .experiments.runner import ExperimentRunner

__all__ = [
    'AdaptiveCurriculumAgent',
    'ClosedLoopSystem',
    'ExperimentRunner',
]
