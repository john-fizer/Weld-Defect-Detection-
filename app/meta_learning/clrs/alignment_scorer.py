"""
Alignment Scorer

Measures alignment between model outputs and user preferences/feedback
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict
import re


class AlignmentScorer:
    """
    Scores alignment of model with user preferences

    Approaches:
    - Feedback correlation: outputs with high feedback
    - Preference consistency: similar inputs get similar outputs
    - Reward modeling: learn what users prefer
    - Behavioral alignment: model follows user patterns
    """

    def __init__(self):
        self.preference_history: List[Dict[str, Any]] = []
        self.learned_preferences: Dict[str, float] = {}

    def calculate_alignment(
        self,
        inputs: List[str],
        outputs: List[str],
        feedback_scores: List[float]
    ) -> float:
        """
        Calculate overall alignment score

        Args:
            inputs: Model inputs
            outputs: Model outputs
            feedback_scores: User feedback (0-1 scale)

        Returns:
            Alignment score (0-1)
        """
        if not outputs or not feedback_scores:
            return 0.5  # Neutral

        # 1. Direct feedback alignment
        feedback_alignment = np.mean(feedback_scores)

        # 2. Consistency alignment
        consistency_score = self._calculate_consistency(inputs, outputs, feedback_scores)

        # 3. Preference alignment (if preferences learned)
        preference_score = self._calculate_preference_alignment(outputs, feedback_scores)

        # Weighted combination
        alignment = (
            0.5 * feedback_alignment +
            0.3 * consistency_score +
            0.2 * preference_score
        )

        # Record in history
        self.preference_history.append({
            'alignment': alignment,
            'feedback_alignment': feedback_alignment,
            'consistency_score': consistency_score,
            'preference_score': preference_score,
            'num_samples': len(outputs)
        })

        # Update learned preferences
        self._update_preferences(outputs, feedback_scores)

        return alignment

    def _calculate_consistency(
        self,
        inputs: List[str],
        outputs: List[str],
        feedback_scores: List[float]
    ) -> float:
        """
        Calculate consistency score

        Similar inputs should get similar treatment if they have similar feedback
        """
        if len(inputs) < 2:
            return 1.0  # Not enough data

        # Group similar inputs
        input_groups = self._group_similar_inputs(inputs, outputs, feedback_scores)

        if not input_groups:
            return 1.0

        # Calculate consistency within groups
        consistency_scores = []

        for group in input_groups:
            if len(group['feedback']) > 1:
                # Feedback should be consistent within group
                feedback_std = np.std(group['feedback'])
                # Lower std = higher consistency
                consistency = 1 - min(feedback_std, 1.0)
                consistency_scores.append(consistency)

        return np.mean(consistency_scores) if consistency_scores else 1.0

    def _group_similar_inputs(
        self,
        inputs: List[str],
        outputs: List[str],
        feedback_scores: List[float],
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Group similar inputs together

        Simple approach: group by first few words or length similarity
        """
        groups = []

        for i, (inp, out, score) in enumerate(zip(inputs, outputs, feedback_scores)):
            # Find matching group
            matched = False

            for group in groups:
                # Simple similarity: check length and first words
                example_input = group['inputs'][0]

                if self._simple_similarity(inp, example_input) > similarity_threshold:
                    group['inputs'].append(inp)
                    group['outputs'].append(out)
                    group['feedback'].append(score)
                    matched = True
                    break

            if not matched:
                # Create new group
                groups.append({
                    'inputs': [inp],
                    'outputs': [out],
                    'feedback': [score]
                })

        return groups

    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on word overlap"""
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _calculate_preference_alignment(
        self,
        outputs: List[str],
        feedback_scores: List[float]
    ) -> float:
        """
        Calculate alignment with learned preferences

        Checks if outputs match previously learned high-value patterns
        """
        if not self.learned_preferences:
            return 0.5  # Neutral if no preferences learned

        # Extract features from outputs
        alignment_scores = []

        for output, feedback in zip(outputs, feedback_scores):
            features = self._extract_features(output)

            # Check against learned preferences
            preference_match = 0.0
            for feature, value in features.items():
                if feature in self.learned_preferences:
                    # Feature value weighted by learned preference
                    preference_match += value * self.learned_preferences[feature]

            # Normalize
            preference_match = min(preference_match, 1.0)
            alignment_scores.append(preference_match)

        return np.mean(alignment_scores) if alignment_scores else 0.5

    def _update_preferences(
        self,
        outputs: List[str],
        feedback_scores: List[float]
    ):
        """
        Update learned preferences based on feedback

        Reinforcement learning approach: features from high-feedback outputs
        are weighted higher
        """
        learning_rate = 0.1

        for output, score in zip(outputs, feedback_scores):
            features = self._extract_features(output)

            # Update preference weights
            for feature, value in features.items():
                if feature not in self.learned_preferences:
                    self.learned_preferences[feature] = 0.5  # Initialize neutral

                # Update based on feedback
                # High feedback = increase preference weight
                target = score  # Feedback is already 0-1
                error = target - self.learned_preferences[feature]

                self.learned_preferences[feature] += learning_rate * error * value

                # Clip to [0, 1]
                self.learned_preferences[feature] = np.clip(
                    self.learned_preferences[feature],
                    0.0,
                    1.0
                )

    def _extract_features(self, output: str) -> Dict[str, float]:
        """
        Extract features from output for preference learning

        Features:
        - Length (normalized)
        - Politeness markers
        - Technical terms
        - Specificity
        - Formatting
        """
        features = {}

        # Length feature
        length = len(output)
        features['length_short'] = 1.0 if length < 50 else 0.0
        features['length_medium'] = 1.0 if 50 <= length < 200 else 0.0
        features['length_long'] = 1.0 if length >= 200 else 0.0

        # Politeness markers
        politeness_words = ['please', 'thank', 'appreciate', 'kindly']
        features['polite'] = sum(1 for word in politeness_words if word in output.lower()) / len(politeness_words)

        # Question marks (asking vs asserting)
        features['has_questions'] = 1.0 if '?' in output else 0.0

        # Bullet points/structure
        features['structured'] = 1.0 if any(marker in output for marker in ['-', '•', '1.', '2.']) else 0.0

        # Technical (numbers, specific terms)
        features['technical'] = len(re.findall(r'\d+', output)) / max(len(output.split()), 1)

        return features

    def get_alignment_trend(self) -> Dict[str, Any]:
        """Get alignment trend over time"""
        if not self.preference_history:
            return {'status': 'no_data'}

        alignments = [h['alignment'] for h in self.preference_history]

        trend_coef = 0.0
        if len(alignments) > 1:
            trend_coef = np.polyfit(range(len(alignments)), alignments, 1)[0]

        return {
            'current_alignment': alignments[-1],
            'average_alignment': np.mean(alignments),
            'max_alignment': max(alignments),
            'min_alignment': min(alignments),
            'trend': 'improving' if trend_coef > 0 else 'degrading',
            'trend_coefficient': trend_coef
        }

    def get_learned_preferences(self) -> Dict[str, float]:
        """Get the learned preference weights"""
        return self.learned_preferences.copy()

    def export_preference_model(self, path: str):
        """Export learned preferences for analysis"""
        import json

        export_data = {
            'learned_preferences': self.learned_preferences,
            'history': self.preference_history,
            'num_updates': len(self.preference_history)
        }

        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Preferences exported to {path}")
