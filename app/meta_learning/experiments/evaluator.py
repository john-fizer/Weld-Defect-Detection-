"""
Model Evaluator

Evaluates prompts and models on datasets
"""

import asyncio
import re
from typing import Dict, List, Optional, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ModelEvaluator:
    """
    Evaluates model performance on datasets

    Supports multiple evaluation metrics:
    - Accuracy
    - Precision/Recall/F1
    - Custom metrics
    """

    def __init__(self):
        self.evaluation_cache: Dict[str, Any] = {}

    async def evaluate_prompt(
        self,
        prompt: str,
        samples: List[Dict[str, Any]],
        llm_client: Optional[Any] = None,
        dataset_name: str = "generic"
    ) -> Dict[str, float]:
        """
        Evaluate a prompt on a dataset

        Args:
            prompt: Prompt to evaluate
            samples: Dataset samples
            llm_client: LLM client for generation
            dataset_name: Name of dataset (for dataset-specific evaluation)

        Returns:
            Dictionary of metrics
        """
        predictions = []
        ground_truth = []
        errors = []

        # Generate predictions for each sample
        for sample in samples:
            # Format prompt with sample
            formatted_prompt = self._format_prompt(prompt, sample, dataset_name)

            # Generate prediction
            if llm_client:
                prediction = await self._generate_with_llm(formatted_prompt, llm_client)
            else:
                # Fallback: rule-based or random for demonstration
                prediction = self._generate_fallback(sample, dataset_name)

            predictions.append(prediction)
            ground_truth.append(sample['expected'])

            # Track errors
            if not self._is_correct(prediction, sample['expected'], dataset_name):
                errors.append({
                    'input': sample['input'],
                    'expected': sample['expected'],
                    'predicted': prediction,
                    'error_type': self._classify_error(prediction, sample['expected'])
                })

        # Calculate metrics
        metrics = self._calculate_metrics(
            predictions,
            ground_truth,
            dataset_name
        )

        # Add error analysis
        metrics['error_rate'] = len(errors) / len(samples) if samples else 0
        metrics['num_errors'] = len(errors)

        return metrics

    def _format_prompt(
        self,
        prompt: str,
        sample: Dict[str, Any],
        dataset_name: str
    ) -> str:
        """Format prompt with sample data"""

        if dataset_name == "commonsense_qa":
            # Replace {question} placeholder
            formatted = prompt.replace("{question}", sample['input'])
        elif dataset_name == "sentiment140":
            # Replace {tweet} placeholder
            formatted = prompt.replace("{tweet}", sample['input'])
        else:
            # Generic: append input
            formatted = f"{prompt}\n\n{sample['input']}"

        return formatted

    async def _generate_with_llm(
        self,
        prompt: str,
        llm_client: Any
    ) -> str:
        """Generate prediction using LLM"""

        try:
            # Anthropic API
            response = await llm_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=100,
                temperature=0.0,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()

        except Exception as e:
            print(f"LLM generation error: {e}")
            return ""

    def _generate_fallback(
        self,
        sample: Dict[str, Any],
        dataset_name: str
    ) -> str:
        """Fallback generation when no LLM available"""

        if dataset_name == "commonsense_qa":
            # Random choice
            choices = sample.get('choices', {})
            if choices:
                import random
                return random.choice(list(choices.keys()))
            return "A"

        elif dataset_name == "sentiment140":
            # Random sentiment
            import random
            return random.choice(['positive', 'negative'])

        else:
            return "unknown"

    def _is_correct(
        self,
        prediction: str,
        expected: str,
        dataset_name: str
    ) -> bool:
        """Check if prediction is correct"""

        # Normalize for comparison
        pred_norm = self._normalize_answer(prediction, dataset_name)
        exp_norm = self._normalize_answer(expected, dataset_name)

        return pred_norm == exp_norm

    def _normalize_answer(self, answer: str, dataset_name: str) -> str:
        """Normalize answer for comparison"""

        if dataset_name == "commonsense_qa":
            # Extract first letter (A, B, C, D, E)
            match = re.search(r'[A-Ea-e]', answer)
            if match:
                return match.group(0).upper()
            return answer.strip().upper()[:1]

        elif dataset_name == "sentiment140":
            # Normalize sentiment
            answer_lower = answer.lower()
            if 'positive' in answer_lower or 'pos' in answer_lower:
                return 'positive'
            elif 'negative' in answer_lower or 'neg' in answer_lower:
                return 'negative'
            return answer_lower.strip()

        else:
            return answer.strip().lower()

    def _classify_error(self, prediction: str, expected: str) -> str:
        """Classify type of error"""

        if not prediction or prediction.strip() == "":
            return "no_response"
        elif len(prediction) > len(expected) * 3:
            return "verbose"
        elif prediction == expected:
            return "none"  # Not actually an error
        else:
            return "incorrect"

    def _calculate_metrics(
        self,
        predictions: List[str],
        ground_truth: List[str],
        dataset_name: str
    ) -> Dict[str, float]:
        """Calculate evaluation metrics"""

        # Normalize all predictions and ground truth
        predictions_norm = [
            self._normalize_answer(p, dataset_name)
            for p in predictions
        ]
        ground_truth_norm = [
            self._normalize_answer(g, dataset_name)
            for g in ground_truth
        ]

        # Binary correctness
        correct = [
            1 if p == g else 0
            for p, g in zip(predictions_norm, ground_truth_norm)
        ]

        accuracy = np.mean(correct)

        # For multi-class classification
        try:
            # Get unique labels
            unique_labels = list(set(ground_truth_norm + predictions_norm))

            if len(unique_labels) > 1:
                precision = precision_score(
                    ground_truth_norm,
                    predictions_norm,
                    labels=unique_labels,
                    average='weighted',
                    zero_division=0
                )
                recall = recall_score(
                    ground_truth_norm,
                    predictions_norm,
                    labels=unique_labels,
                    average='weighted',
                    zero_division=0
                )
                f1 = f1_score(
                    ground_truth_norm,
                    predictions_norm,
                    labels=unique_labels,
                    average='weighted',
                    zero_division=0
                )
            else:
                precision = accuracy
                recall = accuracy
                f1 = accuracy

        except Exception as e:
            print(f"Metric calculation error: {e}")
            precision = accuracy
            recall = accuracy
            f1 = accuracy

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_samples': len(predictions)
        }

    def evaluate_with_metrics(
        self,
        predictions: List[str],
        ground_truth: List[str],
        custom_metrics: Optional[List[callable]] = None
    ) -> Dict[str, float]:
        """
        Evaluate with custom metrics

        Args:
            predictions: Model predictions
            ground_truth: Ground truth labels
            custom_metrics: List of custom metric functions

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Standard metrics
        correct = [1 if p == g else 0 for p, g in zip(predictions, ground_truth)]
        metrics['accuracy'] = np.mean(correct)

        # Custom metrics
        if custom_metrics:
            for metric_fn in custom_metrics:
                try:
                    metric_name = metric_fn.__name__
                    metric_value = metric_fn(predictions, ground_truth)
                    metrics[metric_name] = metric_value
                except Exception as e:
                    print(f"Custom metric error: {e}")

        return metrics

    def batch_evaluate(
        self,
        prompts: List[str],
        samples: List[Dict[str, Any]],
        llm_client: Optional[Any] = None,
        dataset_name: str = "generic"
    ) -> List[Dict[str, float]]:
        """
        Evaluate multiple prompts

        Args:
            prompts: List of prompts to evaluate
            samples: Dataset samples
            llm_client: LLM client
            dataset_name: Dataset name

        Returns:
            List of metric dictionaries
        """
        results = []

        for i, prompt in enumerate(prompts):
            print(f"Evaluating prompt {i+1}/{len(prompts)}...")
            metrics = asyncio.run(
                self.evaluate_prompt(prompt, samples, llm_client, dataset_name)
            )
            results.append(metrics)

        return results

    def compare_prompts(
        self,
        prompt1: str,
        prompt2: str,
        samples: List[Dict[str, Any]],
        llm_client: Optional[Any] = None,
        dataset_name: str = "generic"
    ) -> Dict[str, Any]:
        """
        Compare two prompts

        Returns:
            Comparison results
        """
        metrics1 = asyncio.run(
            self.evaluate_prompt(prompt1, samples, llm_client, dataset_name)
        )
        metrics2 = asyncio.run(
            self.evaluate_prompt(prompt2, samples, llm_client, dataset_name)
        )

        comparison = {
            'prompt1_metrics': metrics1,
            'prompt2_metrics': metrics2,
            'winner': 'prompt1' if metrics1['accuracy'] > metrics2['accuracy'] else 'prompt2',
            'accuracy_difference': metrics1['accuracy'] - metrics2['accuracy']
        }

        return comparison
