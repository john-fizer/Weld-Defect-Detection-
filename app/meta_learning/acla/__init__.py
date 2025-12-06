"""
Adaptive Curriculum Learning Agent (ACLA)

An LLM that rewrites its own training prompts to improve task accuracy
across iterations.

Research Question: Can meta-prompting outperform static fine-tuning?
"""

from .curriculum_agent import AdaptiveCurriculumAgent, CurriculumConfig
from .prompt_evolver import PromptEvolver
from .performance_tracker import PerformanceTracker

__all__ = [
    'AdaptiveCurriculumAgent',
    'CurriculumConfig',
    'PromptEvolver',
    'PerformanceTracker',
]
