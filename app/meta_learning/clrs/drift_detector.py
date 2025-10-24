"""
Drift Detector

Detects and quantifies model behavior drift over training cycles
"""

import numpy as np
from typing import List, Dict, Optional, Any
from collections import Counter
import re


class DriftDetector:
    """
    Detects distribution drift in model outputs

    Methods:
    - Statistical drift (KL divergence, JS divergence)
    - Vocabulary drift
    - Length drift
    - Style drift
    """

    def __init__(self):
        self.baseline_stats: Optional[Dict[str, Any]] = None
        self.history: List[Dict[str, Any]] = []

    def set_baseline(self, baseline_outputs: List[str]):
        """
        Establish baseline distribution

        Args:
            baseline_outputs: Initial model outputs to use as baseline
        """
        self.baseline_stats = self._calculate_statistics(baseline_outputs)
        print(f"Baseline established from {len(baseline_outputs)} outputs")

    def calculate_drift(self, current_outputs: List[str]) -> float:
        """
        Calculate drift score compared to baseline

        Args:
            current_outputs: Current model outputs

        Returns:
            Drift score (0 = no drift, 1 = maximum drift)
        """
        if self.baseline_stats is None:
            print("Warning: No baseline set, setting current as baseline")
            self.set_baseline(current_outputs)
            return 0.0

        current_stats = self._calculate_statistics(current_outputs)

        # Calculate drift across multiple dimensions
        vocab_drift = self._vocabulary_drift(
            self.baseline_stats['vocabulary'],
            current_stats['vocabulary']
        )

        length_drift = self._length_drift(
            self.baseline_stats['avg_length'],
            current_stats['avg_length']
        )

        distribution_drift = self._distribution_drift(
            self.baseline_stats['word_freq'],
            current_stats['word_freq']
        )

        # Weighted combination
        drift_score = (
            0.4 * vocab_drift +
            0.3 * distribution_drift +
            0.3 * length_drift
        )

        # Record in history
        self.history.append({
            'drift_score': drift_score,
            'vocab_drift': vocab_drift,
            'length_drift': length_drift,
            'distribution_drift': distribution_drift,
            'num_outputs': len(current_outputs)
        })

        return drift_score

    def _calculate_statistics(self, outputs: List[str]) -> Dict[str, Any]:
        """Calculate statistical features of outputs"""

        # Tokenize (simple whitespace tokenization)
        all_words = []
        lengths = []

        for output in outputs:
            words = re.findall(r'\w+', output.lower())
            all_words.extend(words)
            lengths.append(len(output))

        # Word frequency
        word_freq = Counter(all_words)
        total_words = len(all_words)

        # Normalize frequencies
        word_freq_normalized = {
            word: count / total_words
            for word, count in word_freq.items()
        }

        # Vocabulary
        vocabulary = set(all_words)

        # Calculate statistics
        stats = {
            'vocabulary': vocabulary,
            'vocab_size': len(vocabulary),
            'word_freq': word_freq_normalized,
            'avg_length': np.mean(lengths) if lengths else 0,
            'std_length': np.std(lengths) if lengths else 0,
            'total_words': total_words,
            'unique_word_ratio': len(vocabulary) / total_words if total_words > 0 else 0
        }

        return stats

    def _vocabulary_drift(self, baseline_vocab: set, current_vocab: set) -> float:
        """
        Calculate vocabulary drift using Jaccard distance

        Args:
            baseline_vocab: Baseline vocabulary set
            current_vocab: Current vocabulary set

        Returns:
            Drift score (0-1)
        """
        if not baseline_vocab or not current_vocab:
            return 0.0

        intersection = len(baseline_vocab & current_vocab)
        union = len(baseline_vocab | current_vocab)

        jaccard_similarity = intersection / union if union > 0 else 0
        jaccard_distance = 1 - jaccard_similarity

        return jaccard_distance

    def _length_drift(self, baseline_length: float, current_length: float) -> float:
        """
        Calculate length drift

        Args:
            baseline_length: Baseline average length
            current_length: Current average length

        Returns:
            Drift score (0-1)
        """
        if baseline_length == 0:
            return 0.0

        # Normalized absolute difference
        diff = abs(current_length - baseline_length)
        drift = min(diff / baseline_length, 1.0)

        return drift

    def _distribution_drift(
        self,
        baseline_freq: Dict[str, float],
        current_freq: Dict[str, float]
    ) -> float:
        """
        Calculate distribution drift using Jensen-Shannon divergence

        Args:
            baseline_freq: Baseline word frequency distribution
            current_freq: Current word frequency distribution

        Returns:
            Drift score (0-1)
        """
        # Get all words
        all_words = set(baseline_freq.keys()) | set(current_freq.keys())

        if not all_words:
            return 0.0

        # Create probability vectors
        baseline_probs = np.array([
            baseline_freq.get(word, 0) for word in all_words
        ])
        current_probs = np.array([
            current_freq.get(word, 0) for word in all_words
        ])

        # Normalize
        baseline_probs = baseline_probs / (baseline_probs.sum() + 1e-10)
        current_probs = current_probs / (current_probs.sum() + 1e-10)

        # Jensen-Shannon divergence
        js_divergence = self._jensen_shannon_divergence(baseline_probs, current_probs)

        return js_divergence

    def _jensen_shannon_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate Jensen-Shannon divergence between two distributions

        Args:
            p: First probability distribution
            q: Second probability distribution

        Returns:
            JS divergence (0-1)
        """
        # Average distribution
        m = (p + q) / 2

        # KL divergences
        kl_pm = self._kl_divergence(p, m)
        kl_qm = self._kl_divergence(q, m)

        # JS divergence
        js = (kl_pm + kl_qm) / 2

        # JS divergence is bounded by log(2), normalize to [0, 1]
        js_normalized = js / np.log(2)

        return min(js_normalized, 1.0)

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate KL divergence"""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon

        return np.sum(p * np.log(p / q))

    def get_drift_trend(self) -> Dict[str, Any]:
        """Get drift trend analysis"""
        if not self.history:
            return {'status': 'no_data'}

        drift_scores = [h['drift_score'] for h in self.history]

        # Calculate trend
        if len(drift_scores) > 1:
            trend_coef = np.polyfit(range(len(drift_scores)), drift_scores, 1)[0]
        else:
            trend_coef = 0.0

        return {
            'current_drift': drift_scores[-1],
            'average_drift': np.mean(drift_scores),
            'max_drift': max(drift_scores),
            'min_drift': min(drift_scores),
            'trend': 'increasing' if trend_coef > 0 else 'decreasing',
            'trend_coefficient': trend_coef,
            'num_measurements': len(drift_scores)
        }

    def detect_sudden_drift(self, threshold: float = 0.2) -> bool:
        """
        Detect sudden drift spikes

        Args:
            threshold: Minimum change to consider a spike

        Returns:
            True if sudden drift detected
        """
        if len(self.history) < 2:
            return False

        recent_drift = self.history[-1]['drift_score']
        previous_drift = self.history[-2]['drift_score']

        drift_change = recent_drift - previous_drift

        return drift_change > threshold

    def get_drift_components(self) -> Dict[str, List[float]]:
        """Get breakdown of drift components over time"""
        return {
            'total_drift': [h['drift_score'] for h in self.history],
            'vocab_drift': [h['vocab_drift'] for h in self.history],
            'length_drift': [h['length_drift'] for h in self.history],
            'distribution_drift': [h['distribution_drift'] for h in self.history]
        }
