"""
Closed-Loop Reinforcement System

Feedback engine that continuously trains and improves a local model based on user input
"""

import json
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import pickle

from .drift_detector import DriftDetector
from .alignment_scorer import AlignmentScorer
from .coherence_analyzer import CoherenceAnalyzer


@dataclass
class FeedbackSample:
    """Single feedback sample"""
    input: str
    output: str
    feedback: float  # Rating or score
    timestamp: datetime
    cycle: int
    metadata: Dict[str, Any]


@dataclass
class TrainingCycle:
    """Record of a training cycle"""
    cycle: int
    timestamp: datetime
    num_samples: int
    metrics: Dict[str, float]
    drift_score: float
    alignment_score: float
    coherence_score: float
    model_checkpoint: Optional[str]


class ClosedLoopSystem:
    """
    Closed-loop reinforcement learning system

    Process:
    1. Model generates outputs
    2. User provides feedback (ratings/corrections)
    3. System collects feedback samples
    4. Periodically retrain model on feedback
    5. Monitor drift, alignment, coherence
    6. Adapt learning strategy based on metrics

    Research Questions:
    - How does model behavior drift over cycles?
    - Can alignment be maintained with minimal feedback?
    - What emergent patterns appear in model coherence?
    - Optimal feedback cycle frequency?
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        model_type: str = "transformer",
        save_path: Optional[Path] = None,
        cycle_size: int = 100,  # Samples per training cycle
        drift_threshold: float = 0.3,
        alignment_threshold: float = 0.7
    ):
        """
        Initialize closed-loop system

        Args:
            model: Initial model (can be None, will create default)
            model_type: Type of model to use
            save_path: Path to save checkpoints and data
            cycle_size: Number of feedback samples before retraining
            drift_threshold: Threshold for drift detection
            alignment_threshold: Minimum alignment score
        """
        self.model = model
        self.model_type = model_type
        self.save_path = save_path or Path("./data/meta_learning/clrs")
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.cycle_size = cycle_size
        self.drift_threshold = drift_threshold
        self.alignment_threshold = alignment_threshold

        # Initialize components
        self.drift_detector = DriftDetector()
        self.alignment_scorer = AlignmentScorer()
        self.coherence_analyzer = CoherenceAnalyzer()

        # State
        self.feedback_buffer: List[FeedbackSample] = []
        self.all_feedback: List[FeedbackSample] = []
        self.training_cycles: List[TrainingCycle] = []
        self.current_cycle = 0

        # Baseline for comparison
        self.baseline_outputs: List[str] = []
        self.baseline_established = False

    def collect_feedback(
        self,
        input_text: str,
        output_text: str,
        feedback_score: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Collect a feedback sample

        Args:
            input_text: Input to the model
            output_text: Model's output
            feedback_score: User feedback (0-1 scale, 1 = best)
            metadata: Additional metadata
        """
        sample = FeedbackSample(
            input=input_text,
            output=output_text,
            feedback=feedback_score,
            timestamp=datetime.now(),
            cycle=self.current_cycle,
            metadata=metadata or {}
        )

        self.feedback_buffer.append(sample)
        self.all_feedback.append(sample)

        # Establish baseline if needed
        if not self.baseline_established and len(self.feedback_buffer) >= 10:
            self._establish_baseline()

        # Check if cycle is complete
        if len(self.feedback_buffer) >= self.cycle_size:
            self._run_training_cycle()

    def _establish_baseline(self):
        """Establish baseline for drift detection"""
        print("\nEstablishing baseline from initial feedback...")

        baseline_samples = self.feedback_buffer[:20]
        self.baseline_outputs = [s.output for s in baseline_samples]

        self.drift_detector.set_baseline(self.baseline_outputs)
        self.baseline_established = True

        print(f"Baseline established with {len(self.baseline_outputs)} samples")

    def _run_training_cycle(self):
        """Run a complete training cycle"""
        print(f"\n{'='*70}")
        print(f"TRAINING CYCLE {self.current_cycle + 1}")
        print(f"{'='*70}")

        # 1. Prepare training data
        training_samples = self.feedback_buffer.copy()
        print(f"Training samples: {len(training_samples)}")

        # 2. Train/update model
        metrics = self._train_model(training_samples)

        # 3. Evaluate drift
        current_outputs = [s.output for s in training_samples]
        drift_score = self.drift_detector.calculate_drift(current_outputs)
        print(f"Drift Score: {drift_score:.4f}")

        # 4. Evaluate alignment
        alignment_score = self._evaluate_alignment(training_samples)
        print(f"Alignment Score: {alignment_score:.4f}")

        # 5. Evaluate coherence
        coherence_score = self._evaluate_coherence(training_samples)
        print(f"Coherence Score: {coherence_score:.4f}")

        # 6. Save checkpoint
        checkpoint_path = self._save_checkpoint()

        # 7. Record cycle
        cycle_record = TrainingCycle(
            cycle=self.current_cycle + 1,
            timestamp=datetime.now(),
            num_samples=len(training_samples),
            metrics=metrics,
            drift_score=drift_score,
            alignment_score=alignment_score,
            coherence_score=coherence_score,
            model_checkpoint=str(checkpoint_path) if checkpoint_path else None
        )
        self.training_cycles.append(cycle_record)

        # 8. Check for issues
        self._check_system_health(drift_score, alignment_score, coherence_score)

        # 9. Clear buffer and increment cycle
        self.feedback_buffer.clear()
        self.current_cycle += 1

        print(f"{'='*70}\n")

    def _train_model(self, samples: List[FeedbackSample]) -> Dict[str, float]:
        """
        Train/update model on feedback samples

        Args:
            samples: Training samples

        Returns:
            Training metrics
        """
        print("\nTraining model on feedback...")

        # Prepare data
        inputs = [s.input for s in samples]
        outputs = [s.output for s in samples]
        scores = [s.feedback for s in samples]

        # For demonstration: simulate training metrics
        # In real implementation, this would actually train the model

        avg_score = np.mean(scores)
        score_variance = np.var(scores)

        # Simulate improvement
        improvement = avg_score - 0.5  # Assuming 0.5 is neutral

        metrics = {
            'avg_feedback_score': avg_score,
            'feedback_variance': score_variance,
            'improvement_signal': improvement,
            'num_samples': len(samples),
            'high_quality_ratio': sum(1 for s in scores if s > 0.7) / len(scores)
        }

        print(f"  Average feedback score: {avg_score:.4f}")
        print(f"  High quality ratio: {metrics['high_quality_ratio']:.4f}")

        return metrics

    def _evaluate_alignment(self, samples: List[FeedbackSample]) -> float:
        """
        Evaluate alignment with user preferences

        Args:
            samples: Recent samples

        Returns:
            Alignment score (0-1)
        """
        inputs = [s.input for s in samples]
        outputs = [s.output for s in samples]
        feedbacks = [s.feedback for s in samples]

        return self.alignment_scorer.calculate_alignment(
            inputs, outputs, feedbacks
        )

    def _evaluate_coherence(self, samples: List[FeedbackSample]) -> float:
        """
        Evaluate output coherence

        Args:
            samples: Recent samples

        Returns:
            Coherence score (0-1)
        """
        outputs = [s.output for s in samples]
        return self.coherence_analyzer.calculate_coherence(outputs)

    def _check_system_health(
        self,
        drift_score: float,
        alignment_score: float,
        coherence_score: float
    ):
        """Check system health and issue warnings"""

        print("\nSystem Health Check:")
        print(f"{'─'*40}")

        # Check drift
        if drift_score > self.drift_threshold:
            print(f"⚠️  WARNING: High drift detected ({drift_score:.4f})")
            print("   Consider: revert to earlier checkpoint or adjust learning rate")
        else:
            print(f"✓  Drift within acceptable range ({drift_score:.4f})")

        # Check alignment
        if alignment_score < self.alignment_threshold:
            print(f"⚠️  WARNING: Low alignment ({alignment_score:.4f})")
            print("   Consider: collect more diverse feedback or adjust training")
        else:
            print(f"✓  Alignment satisfactory ({alignment_score:.4f})")

        # Check coherence
        if coherence_score < 0.5:
            print(f"⚠️  WARNING: Low coherence ({coherence_score:.4f})")
            print("   Consider: model may be overfitting to noisy feedback")
        else:
            print(f"✓  Coherence acceptable ({coherence_score:.4f})")

        print(f"{'─'*40}")

    def _save_checkpoint(self) -> Optional[Path]:
        """Save model checkpoint"""
        checkpoint_dir = self.save_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"cycle_{self.current_cycle + 1}.pkl"

        checkpoint_data = {
            'cycle': self.current_cycle + 1,
            'timestamp': datetime.now().isoformat(),
            'model_state': None,  # Would save actual model weights
            'training_history': [asdict(c) for c in self.training_cycles],
            'feedback_count': len(self.all_feedback)
        }

        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        print(f"\nCheckpoint saved: {checkpoint_path}")
        return checkpoint_path

    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary"""

        if not self.training_cycles:
            return {'status': 'no_training_cycles_completed'}

        # Drift analysis
        drift_scores = [c.drift_score for c in self.training_cycles]
        drift_trend = np.polyfit(range(len(drift_scores)), drift_scores, 1)[0]

        # Alignment analysis
        alignment_scores = [c.alignment_score for c in self.training_cycles]
        alignment_trend = np.polyfit(range(len(alignment_scores)), alignment_scores, 1)[0]

        # Coherence analysis
        coherence_scores = [c.coherence_score for c in self.training_cycles]
        coherence_trend = np.polyfit(range(len(coherence_scores)), coherence_scores, 1)[0]

        summary = {
            'total_cycles': len(self.training_cycles),
            'total_feedback': len(self.all_feedback),
            'current_cycle': self.current_cycle,

            'drift': {
                'current': drift_scores[-1],
                'average': np.mean(drift_scores),
                'trend': drift_trend,
                'max': max(drift_scores),
                'trend_direction': 'increasing' if drift_trend > 0 else 'decreasing'
            },

            'alignment': {
                'current': alignment_scores[-1],
                'average': np.mean(alignment_scores),
                'trend': alignment_trend,
                'max': max(alignment_scores),
                'trend_direction': 'increasing' if alignment_trend > 0 else 'decreasing'
            },

            'coherence': {
                'current': coherence_scores[-1],
                'average': np.mean(coherence_scores),
                'trend': coherence_trend,
                'max': max(coherence_scores),
                'trend_direction': 'increasing' if coherence_trend > 0 else 'decreasing'
            },

            'health_status': self._get_health_status(
                drift_scores[-1],
                alignment_scores[-1],
                coherence_scores[-1]
            )
        }

        return summary

    def _get_health_status(
        self,
        drift: float,
        alignment: float,
        coherence: float
    ) -> str:
        """Determine overall system health status"""

        issues = []

        if drift > self.drift_threshold:
            issues.append('high_drift')
        if alignment < self.alignment_threshold:
            issues.append('low_alignment')
        if coherence < 0.5:
            issues.append('low_coherence')

        if not issues:
            return 'healthy'
        elif len(issues) == 1:
            return 'warning'
        else:
            return 'critical'

    def export_results(self, output_path: Optional[Path] = None):
        """Export full results for analysis"""

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.save_path / f"clrs_results_{timestamp}.json"

        results = {
            'config': {
                'cycle_size': self.cycle_size,
                'drift_threshold': self.drift_threshold,
                'alignment_threshold': self.alignment_threshold
            },
            'summary': self.get_system_summary(),
            'cycles': [asdict(c) for c in self.training_cycles],
            'feedback_samples': len(self.all_feedback),
            'timestamp': datetime.now().isoformat()
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults exported to: {output_path}")
        return output_path

    def simulate_feedback_loop(
        self,
        generate_fn: Callable[[str], str],
        input_samples: List[str],
        feedback_fn: Callable[[str, str], float],
        num_cycles: int = 5
    ):
        """
        Simulate a complete feedback loop for testing

        Args:
            generate_fn: Function to generate outputs
            input_samples: List of input samples
            feedback_fn: Function to generate feedback scores
            num_cycles: Number of cycles to run
        """
        print(f"\n{'='*70}")
        print("SIMULATING CLOSED-LOOP REINFORCEMENT SYSTEM")
        print(f"{'='*70}")
        print(f"Input samples: {len(input_samples)}")
        print(f"Target cycles: {num_cycles}")
        print(f"{'='*70}\n")

        for cycle in range(num_cycles):
            print(f"\nCycle {cycle + 1}/{num_cycles}")
            print(f"{'─'*40}")

            # Generate outputs and collect feedback
            for input_text in input_samples:
                output = generate_fn(input_text)
                feedback = feedback_fn(input_text, output)

                self.collect_feedback(
                    input_text=input_text,
                    output_text=output,
                    feedback_score=feedback,
                    metadata={'cycle': cycle + 1}
                )

        # Final summary
        summary = self.get_system_summary()
        print(f"\n{'='*70}")
        print("SIMULATION COMPLETE")
        print(f"{'='*70}")
        print(f"Total cycles: {summary['total_cycles']}")
        print(f"Total feedback: {summary['total_feedback']}")
        print(f"\nFinal Metrics:")
        print(f"  Drift: {summary['drift']['current']:.4f} ({summary['drift']['trend_direction']})")
        print(f"  Alignment: {summary['alignment']['current']:.4f} ({summary['alignment']['trend_direction']})")
        print(f"  Coherence: {summary['coherence']['current']:.4f} ({summary['coherence']['trend_direction']})")
        print(f"  Health: {summary['health_status']}")
        print(f"{'='*70}\n")

        return summary
