"""
Closed-Loop Reinforcement System (CLRS)

Self-optimizing feedback engine where user input trains a smaller local model.
Studies drift, alignment, and emergent coherence over multiple feedback cycles.
"""

from .reinforcement_system import ClosedLoopSystem
from .drift_detector import DriftDetector
from .alignment_scorer import AlignmentScorer
from .coherence_analyzer import CoherenceAnalyzer

__all__ = [
    'ClosedLoopSystem',
    'DriftDetector',
    'AlignmentScorer',
    'CoherenceAnalyzer',
]
