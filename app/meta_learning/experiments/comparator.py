"""
Experiment Comparator

Compares results from different experiments
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


class ExperimentComparator:
    """
    Compare experimental results

    Focus:
    - Meta-prompting vs fine-tuning
    - Different evolution strategies
    - System configurations
    """

    def compare_experiments(
        self,
        experiment1: Any,  # ExperimentResult
        experiment2: Any   # ExperimentResult
    ) -> Dict[str, Any]:
        """
        Compare two experiments

        Args:
            experiment1: First experiment (typically meta-prompting)
            experiment2: Second experiment (typically baseline)

        Returns:
            Comparison analysis
        """
        # Extract key metrics
        exp1_final = experiment1.final_performance.get('accuracy', 0)
        exp2_final = experiment2.final_performance.get('accuracy', 0)

        exp1_best = experiment1.best_performance.get('accuracy', 0)
        exp2_best = experiment2.best_performance.get('accuracy', 0)

        exp1_improvement = experiment1.improvement
        exp2_improvement = experiment2.improvement

        # Determine winner
        meta_prompting_wins = exp1_final > exp2_final
        win_margin = abs(exp1_final - exp2_final)

        # Statistical significance (simplified)
        significance = self._calculate_significance(
            experiment1.performance_history,
            experiment2.performance_history
        )

        # Learning efficiency
        efficiency = self._calculate_efficiency(
            experiment1.performance_history,
            experiment2.performance_history
        )

        comparison = {
            'meta_prompting_wins': meta_prompting_wins,
            'win_margin': win_margin,

            'final_performance': {
                'meta_prompting': exp1_final,
                'baseline': exp2_final,
                'difference': exp1_final - exp2_final,
                'relative_improvement': (exp1_final - exp2_final) / exp2_final if exp2_final > 0 else 0
            },

            'best_performance': {
                'meta_prompting': exp1_best,
                'baseline': exp2_best,
                'difference': exp1_best - exp2_best
            },

            'improvement': {
                'meta_prompting': exp1_improvement,
                'baseline': exp2_improvement,
                'difference': exp1_improvement - exp2_improvement
            },

            'convergence': {
                'meta_prompting_iterations': experiment1.convergence_iteration,
                'baseline_iterations': experiment2.convergence_iteration,
                'meta_prompting_faster': experiment1.convergence_iteration < experiment2.convergence_iteration
            },

            'learning_efficiency': efficiency,
            'statistical_significance': significance,

            'conclusion': self._generate_conclusion(
                meta_prompting_wins,
                win_margin,
                significance
            )
        }

        return comparison

    def _calculate_significance(
        self,
        history1: List[Dict[str, float]],
        history2: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Calculate statistical significance of difference

        Simplified approach using variance and sample size
        """
        # Extract accuracy trends
        acc1 = [h.get('accuracy', 0) for h in history1]
        acc2 = [h.get('accuracy', 0) for h in history2]

        # Calculate statistics
        mean1 = np.mean(acc1)
        mean2 = np.mean(acc2)
        std1 = np.std(acc1)
        std2 = np.std(acc2)

        # Simple significance test (t-test approximation)
        n1 = len(acc1)
        n2 = len(acc2)

        if std1 == 0 and std2 == 0:
            # No variance
            significant = mean1 != mean2
        else:
            # Pooled standard error
            se = np.sqrt((std1**2 / n1) + (std2**2 / n2))

            if se > 0:
                t_statistic = abs(mean1 - mean2) / se

                # Rough significance threshold (t > 2 approximately p < 0.05 for large df)
                significant = t_statistic > 2
            else:
                significant = mean1 != mean2

        return {
            'significant': significant,
            'mean_difference': mean1 - mean2,
            'std1': std1,
            'std2': std2,
            'effect_size': abs(mean1 - mean2) / max(std1, std2, 0.01)
        }

    def _calculate_efficiency(
        self,
        history1: List[Dict[str, float]],
        history2: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Calculate learning efficiency

        Measures how quickly performance improves
        """
        # Extract accuracy trends
        acc1 = [h.get('accuracy', 0) for h in history1]
        acc2 = [h.get('accuracy', 0) for h in history2]

        # Calculate improvement rates
        if len(acc1) > 1:
            improvements1 = [acc1[i] - acc1[i-1] for i in range(1, len(acc1))]
            avg_improvement1 = np.mean(improvements1)
            early_improvement1 = np.mean(improvements1[:3]) if len(improvements1) >= 3 else np.mean(improvements1)
        else:
            avg_improvement1 = 0
            early_improvement1 = 0

        if len(acc2) > 1:
            improvements2 = [acc2[i] - acc2[i-1] for i in range(1, len(acc2))]
            avg_improvement2 = np.mean(improvements2)
            early_improvement2 = np.mean(improvements2[:3]) if len(improvements2) >= 3 else np.mean(improvements2)
        else:
            avg_improvement2 = 0
            early_improvement2 = 0

        return {
            'meta_prompting_avg_improvement': avg_improvement1,
            'baseline_avg_improvement': avg_improvement2,
            'meta_prompting_early_improvement': early_improvement1,
            'baseline_early_improvement': early_improvement2,
            'meta_prompting_more_efficient': avg_improvement1 > avg_improvement2
        }

    def _generate_conclusion(
        self,
        meta_wins: bool,
        margin: float,
        significance: Dict[str, Any]
    ) -> str:
        """Generate human-readable conclusion"""

        if meta_wins:
            if significance['significant']:
                return f"Meta-prompting SIGNIFICANTLY outperforms static baseline (margin: {margin:.4f}, significant: yes)"
            else:
                return f"Meta-prompting outperforms static baseline (margin: {margin:.4f}, but not statistically significant)"
        else:
            if significance['significant']:
                return f"Static baseline SIGNIFICANTLY outperforms meta-prompting (margin: {margin:.4f})"
            else:
                return f"Results are comparable (margin: {margin:.4f}, not significant)"

    def multi_experiment_comparison(
        self,
        experiments: List[Any]  # List of ExperimentResults
    ) -> Dict[str, Any]:
        """
        Compare multiple experiments

        Args:
            experiments: List of experiment results

        Returns:
            Multi-way comparison
        """
        if not experiments:
            return {'status': 'no_experiments'}

        # Extract metrics for all experiments
        results = []

        for exp in experiments:
            results.append({
                'name': exp.config.name,
                'approach': exp.config.approach,
                'final_accuracy': exp.final_performance.get('accuracy', 0),
                'best_accuracy': exp.best_performance.get('accuracy', 0),
                'improvement': exp.improvement,
                'convergence_iteration': exp.convergence_iteration
            })

        # Find best
        best_exp = max(results, key=lambda r: r['final_accuracy'])

        # Rankings
        results_sorted = sorted(results, key=lambda r: r['final_accuracy'], reverse=True)

        return {
            'num_experiments': len(experiments),
            'results': results_sorted,
            'best_experiment': best_exp,
            'approach_comparison': self._compare_approaches(results)
        }

    def _compare_approaches(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare different approaches (meta-prompting, baseline, etc.)"""

        # Group by approach
        by_approach = {}

        for result in results:
            approach = result['approach']
            if approach not in by_approach:
                by_approach[approach] = []
            by_approach[approach].append(result)

        # Calculate stats per approach
        approach_stats = {}

        for approach, approach_results in by_approach.items():
            accuracies = [r['final_accuracy'] for r in approach_results]

            approach_stats[approach] = {
                'count': len(approach_results),
                'avg_accuracy': np.mean(accuracies),
                'max_accuracy': max(accuracies),
                'min_accuracy': min(accuracies),
                'std_accuracy': np.std(accuracies)
            }

        return approach_stats
