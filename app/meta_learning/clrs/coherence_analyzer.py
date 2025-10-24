"""
Coherence Analyzer

Analyzes emergent coherence patterns in model outputs over training cycles
"""

import numpy as np
from typing import List, Dict, Optional, Any
import re
from collections import Counter, defaultdict


class CoherenceAnalyzer:
    """
    Analyzes output coherence

    Measures:
    - Internal consistency
    - Logical flow
    - Semantic coherence
    - Emergent patterns
    """

    def __init__(self):
        self.coherence_history: List[Dict[str, Any]] = []

    def calculate_coherence(self, outputs: List[str]) -> float:
        """
        Calculate coherence score for a batch of outputs

        Args:
            outputs: Model outputs

        Returns:
            Coherence score (0-1)
        """
        if not outputs:
            return 0.0

        # Multiple coherence dimensions
        lexical_coherence = self._lexical_coherence(outputs)
        length_coherence = self._length_coherence(outputs)
        pattern_coherence = self._pattern_coherence(outputs)
        diversity_coherence = self._diversity_coherence(outputs)

        # Weighted combination
        coherence = (
            0.3 * lexical_coherence +
            0.2 * length_coherence +
            0.3 * pattern_coherence +
            0.2 * diversity_coherence
        )

        # Record in history
        self.coherence_history.append({
            'coherence': coherence,
            'lexical_coherence': lexical_coherence,
            'length_coherence': length_coherence,
            'pattern_coherence': pattern_coherence,
            'diversity_coherence': diversity_coherence,
            'num_outputs': len(outputs)
        })

        return coherence

    def _lexical_coherence(self, outputs: List[str]) -> float:
        """
        Lexical coherence: consistent vocabulary usage

        High coherence = stable vocabulary distribution
        """
        if len(outputs) < 2:
            return 1.0

        # Calculate word distributions for each output
        distributions = []

        for output in outputs:
            words = re.findall(r'\w+', output.lower())
            word_count = Counter(words)
            total = len(words)

            if total > 0:
                dist = {word: count/total for word, count in word_count.items()}
                distributions.append(dist)

        if len(distributions) < 2:
            return 1.0

        # Calculate average pairwise similarity
        similarities = []

        for i in range(len(distributions)):
            for j in range(i+1, len(distributions)):
                sim = self._distribution_similarity(
                    distributions[i],
                    distributions[j]
                )
                similarities.append(sim)

        return np.mean(similarities) if similarities else 0.5

    def _distribution_similarity(
        self,
        dist1: Dict[str, float],
        dist2: Dict[str, float]
    ) -> float:
        """Calculate similarity between two word distributions"""

        all_words = set(dist1.keys()) | set(dist2.keys())

        if not all_words:
            return 0.0

        # Cosine similarity
        vec1 = np.array([dist1.get(word, 0) for word in all_words])
        vec2 = np.array([dist2.get(word, 0) for word in all_words])

        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        return similarity

    def _length_coherence(self, outputs: List[str]) -> float:
        """
        Length coherence: consistent output lengths

        High coherence = low variance in length
        """
        if len(outputs) < 2:
            return 1.0

        lengths = [len(output) for output in outputs]
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)

        if mean_length == 0:
            return 0.0

        # Coefficient of variation
        cv = std_length / mean_length

        # Convert to coherence score (lower CV = higher coherence)
        coherence = max(0, 1 - cv)

        return coherence

    def _pattern_coherence(self, outputs: List[str]) -> float:
        """
        Pattern coherence: consistent structural patterns

        Looks for:
        - Sentence structure
        - Punctuation patterns
        - Formatting patterns
        """
        if len(outputs) < 2:
            return 1.0

        patterns = []

        for output in outputs:
            pattern_features = {
                'num_sentences': len(re.split(r'[.!?]+', output)),
                'num_questions': output.count('?'),
                'num_exclamations': output.count('!'),
                'has_bullets': 1 if any(m in output for m in ['-', '•', '*']) else 0,
                'has_numbers': 1 if re.search(r'\d+', output) else 0,
                'has_quotes': 1 if '"' in output or "'" in output else 0
            }
            patterns.append(pattern_features)

        # Calculate consistency of each feature
        feature_coherences = []

        for feature in patterns[0].keys():
            values = [p[feature] for p in patterns]
            # Low variance = high coherence
            if np.std(values) == 0:
                feature_coherences.append(1.0)
            else:
                mean_val = np.mean(values)
                if mean_val > 0:
                    cv = np.std(values) / mean_val
                    coherence = max(0, 1 - cv)
                else:
                    coherence = 1.0
                feature_coherences.append(coherence)

        return np.mean(feature_coherences) if feature_coherences else 0.5

    def _diversity_coherence(self, outputs: List[str]) -> float:
        """
        Diversity coherence: balanced between repetition and variety

        Too repetitive = low coherence (mode collapse)
        Too diverse = low coherence (instability)
        """
        if len(outputs) < 2:
            return 1.0

        # Check for exact duplicates
        unique_outputs = len(set(outputs))
        duplicate_ratio = unique_outputs / len(outputs)

        # Check for near-duplicates (high overlap)
        similarities = []
        for i in range(len(outputs)):
            for j in range(i+1, len(outputs)):
                sim = self._text_similarity(outputs[i], outputs[j])
                similarities.append(sim)

        avg_similarity = np.mean(similarities) if similarities else 0.5

        # Ideal: moderate similarity (not too high, not too low)
        # Bell curve centered at 0.5
        ideal_similarity = 0.5
        similarity_coherence = 1 - abs(avg_similarity - ideal_similarity)

        # Penalize exact duplicates
        duplicate_penalty = duplicate_ratio

        coherence = similarity_coherence * duplicate_penalty

        return coherence

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using word overlap"""
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def detect_mode_collapse(
        self,
        outputs: List[str],
        threshold: float = 0.9
    ) -> bool:
        """
        Detect mode collapse (repetitive outputs)

        Args:
            outputs: Model outputs
            threshold: Similarity threshold for collapse detection

        Returns:
            True if mode collapse detected
        """
        if len(outputs) < 3:
            return False

        # Check pairwise similarities
        high_similarity_count = 0
        total_pairs = 0

        for i in range(len(outputs)):
            for j in range(i+1, len(outputs)):
                sim = self._text_similarity(outputs[i], outputs[j])
                if sim > threshold:
                    high_similarity_count += 1
                total_pairs += 1

        # If more than 50% of pairs are highly similar
        collapse_ratio = high_similarity_count / total_pairs if total_pairs > 0 else 0

        return collapse_ratio > 0.5

    def get_coherence_trend(self) -> Dict[str, Any]:
        """Get coherence trend analysis"""
        if not self.coherence_history:
            return {'status': 'no_data'}

        coherence_scores = [h['coherence'] for h in self.coherence_history]

        trend_coef = 0.0
        if len(coherence_scores) > 1:
            trend_coef = np.polyfit(range(len(coherence_scores)), coherence_scores, 1)[0]

        return {
            'current_coherence': coherence_scores[-1],
            'average_coherence': np.mean(coherence_scores),
            'max_coherence': max(coherence_scores),
            'min_coherence': min(coherence_scores),
            'trend': 'improving' if trend_coef > 0 else 'degrading',
            'trend_coefficient': trend_coef,
            'volatility': np.std(coherence_scores)
        }

    def analyze_emergent_patterns(self, outputs: List[str]) -> Dict[str, Any]:
        """
        Analyze emergent patterns in outputs

        Returns:
            Dictionary of detected patterns and their frequencies
        """
        # Common patterns to detect
        patterns = {
            'greeting_pattern': r'^(hello|hi|hey)',
            'closing_pattern': r'(thanks|thank you|regards)$',
            'question_pattern': r'\?',
            'list_pattern': r'(\n-|\n\d+\.)',
            'code_pattern': r'```|`',
            'emphasis_pattern': r'\*\*|__',
        }

        pattern_counts = defaultdict(int)

        for output in outputs:
            output_lower = output.lower()
            for pattern_name, pattern_regex in patterns.items():
                if re.search(pattern_regex, output_lower):
                    pattern_counts[pattern_name] += 1

        # Calculate frequencies
        total = len(outputs)
        pattern_frequencies = {
            name: count / total
            for name, count in pattern_counts.items()
        }

        # Detect emergent patterns (high frequency)
        emergent = {
            name: freq
            for name, freq in pattern_frequencies.items()
            if freq > 0.7  # Appears in >70% of outputs
        }

        return {
            'pattern_frequencies': pattern_frequencies,
            'emergent_patterns': emergent,
            'num_outputs_analyzed': total
        }
