"""
Performance Tracker

Tracks and analyzes performance metrics across curriculum iterations
"""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
import json
from pathlib import Path


class PerformanceTracker:
    """
    Tracks performance metrics across curriculum learning iterations

    Provides:
    - Historical performance tracking
    - Trend analysis
    - Statistical summaries
    - Improvement rate calculations
    """

    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.metrics_by_iteration: Dict[int, Dict[str, float]] = {}

    def record_iteration(self, iteration: int, metrics: Dict[str, float]):
        """
        Record metrics for an iteration

        Args:
            iteration: Iteration number
            metrics: Dictionary of metric_name -> value
        """
        self.metrics_by_iteration[iteration] = metrics
        self.history.append({
            'iteration': iteration,
            'metrics': metrics.copy()
        })

    def get_history(self) -> List[Dict[str, Any]]:
        """Get full history of recorded metrics"""
        return self.history

    def get_metric_trend(self, metric_name: str) -> List[float]:
        """
        Get trend for a specific metric across iterations

        Args:
            metric_name: Name of metric to track

        Returns:
            List of metric values in iteration order
        """
        trend = []
        for iteration in sorted(self.metrics_by_iteration.keys()):
            value = self.metrics_by_iteration[iteration].get(metric_name, np.nan)
            trend.append(value)
        return trend

    def calculate_improvement_rate(self, metric_name: str) -> float:
        """
        Calculate average improvement rate per iteration

        Args:
            metric_name: Metric to analyze

        Returns:
            Average improvement per iteration
        """
        trend = self.get_metric_trend(metric_name)
        if len(trend) < 2:
            return 0.0

        improvements = [
            trend[i] - trend[i-1]
            for i in range(1, len(trend))
            if not np.isnan(trend[i]) and not np.isnan(trend[i-1])
        ]

        return np.mean(improvements) if improvements else 0.0

    def get_best_iteration(self, metric_name: str) -> Optional[int]:
        """
        Get iteration number with best performance for a metric

        Args:
            metric_name: Metric to optimize

        Returns:
            Iteration number with highest value
        """
        best_iter = None
        best_value = -np.inf

        for iteration, metrics in self.metrics_by_iteration.items():
            value = metrics.get(metric_name, -np.inf)
            if value > best_value:
                best_value = value
                best_iter = iteration

        return best_iter

    def get_convergence_analysis(self, metric_name: str, window: int = 3) -> Dict[str, Any]:
        """
        Analyze convergence pattern for a metric

        Args:
            metric_name: Metric to analyze
            window: Window size for convergence detection

        Returns:
            Convergence analysis dictionary
        """
        trend = self.get_metric_trend(metric_name)

        if len(trend) < window:
            return {
                'converged': False,
                'convergence_iteration': None,
                'is_improving': len(trend) > 1 and trend[-1] > trend[0],
                'volatility': 0.0
            }

        # Calculate rolling improvements
        improvements = [
            trend[i] - trend[i-1]
            for i in range(1, len(trend))
        ]

        # Check for convergence (small improvements in recent window)
        recent_improvements = improvements[-window:]
        avg_recent_improvement = np.mean(np.abs(recent_improvements))
        converged = avg_recent_improvement < 0.01  # Threshold

        # Find convergence point
        convergence_iter = None
        if converged:
            for i in range(len(improvements) - window + 1):
                window_improvements = improvements[i:i+window]
                if np.mean(np.abs(window_improvements)) < 0.01:
                    convergence_iter = i + window
                    break

        # Volatility
        volatility = np.std(improvements) if improvements else 0.0

        return {
            'converged': converged,
            'convergence_iteration': convergence_iter,
            'is_improving': improvements[-1] > 0 if improvements else False,
            'volatility': volatility,
            'avg_improvement': np.mean(improvements) if improvements else 0.0,
            'total_improvement': trend[-1] - trend[0] if len(trend) > 1 else 0.0
        }

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics across all metrics and iterations

        Returns:
            Dictionary of summary statistics
        """
        if not self.history:
            return {'status': 'no_data'}

        # Collect all metric names
        all_metrics = set()
        for record in self.history:
            all_metrics.update(record['metrics'].keys())

        summary = {
            'total_iterations': len(self.history),
            'metrics': {}
        }

        for metric_name in all_metrics:
            trend = self.get_metric_trend(metric_name)
            valid_trend = [v for v in trend if not np.isnan(v)]

            if not valid_trend:
                continue

            metric_summary = {
                'initial': valid_trend[0],
                'final': valid_trend[-1],
                'best': max(valid_trend),
                'worst': min(valid_trend),
                'mean': np.mean(valid_trend),
                'std': np.std(valid_trend),
                'improvement': valid_trend[-1] - valid_trend[0],
                'improvement_rate': self.calculate_improvement_rate(metric_name),
                'convergence': self.get_convergence_analysis(metric_name)
            }

            summary['metrics'][metric_name] = metric_summary

        return summary

    def compare_to_baseline(
        self,
        baseline_metrics: Dict[str, float],
        current_iteration: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compare current performance to baseline (e.g., static fine-tuning)

        Args:
            baseline_metrics: Baseline performance metrics
            current_iteration: Iteration to compare (default: latest)

        Returns:
            Comparison results
        """
        if current_iteration is None:
            current_iteration = max(self.metrics_by_iteration.keys())

        current_metrics = self.metrics_by_iteration.get(current_iteration, {})

        comparison = {
            'iteration': current_iteration,
            'comparison': {}
        }

        for metric_name, baseline_value in baseline_metrics.items():
            current_value = current_metrics.get(metric_name, np.nan)

            if np.isnan(current_value):
                continue

            improvement = current_value - baseline_value
            percent_improvement = (improvement / baseline_value * 100) if baseline_value != 0 else 0

            comparison['comparison'][metric_name] = {
                'baseline': baseline_value,
                'current': current_value,
                'improvement': improvement,
                'percent_improvement': percent_improvement,
                'better': current_value > baseline_value
            }

        # Overall verdict
        better_count = sum(
            1 for m in comparison['comparison'].values()
            if m['better']
        )
        total_count = len(comparison['comparison'])

        comparison['summary'] = {
            'metrics_better': better_count,
            'metrics_total': total_count,
            'better_rate': better_count / total_count if total_count > 0 else 0,
            'overall_better': better_count > total_count / 2
        }

        return comparison

    def export_trends(self, output_path: Path):
        """
        Export trend data for visualization

        Args:
            output_path: Path to save trend data
        """
        trends = {}

        # Get all metrics
        all_metrics = set()
        for record in self.history:
            all_metrics.update(record['metrics'].keys())

        # Export trend for each metric
        for metric_name in all_metrics:
            trends[metric_name] = self.get_metric_trend(metric_name)

        export_data = {
            'iterations': list(sorted(self.metrics_by_iteration.keys())),
            'trends': trends,
            'summary': self.get_summary_statistics()
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Trends exported to {output_path}")

    def plot_trends(self, metric_names: Optional[List[str]] = None, save_path: Optional[Path] = None):
        """
        Plot performance trends (requires matplotlib)

        Args:
            metric_names: List of metrics to plot (None = all)
            save_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting")
            return

        if metric_names is None:
            # Get all metric names
            metric_names = set()
            for record in self.history:
                metric_names.update(record['metrics'].keys())
            metric_names = list(metric_names)

        iterations = list(sorted(self.metrics_by_iteration.keys()))

        fig, axes = plt.subplots(len(metric_names), 1, figsize=(10, 4 * len(metric_names)))

        if len(metric_names) == 1:
            axes = [axes]

        for i, metric_name in enumerate(metric_names):
            trend = self.get_metric_trend(metric_name)

            axes[i].plot(iterations, trend, marker='o', linewidth=2)
            axes[i].set_xlabel('Iteration')
            axes[i].set_ylabel(metric_name)
            axes[i].set_title(f'{metric_name} Trend')
            axes[i].grid(True, alpha=0.3)

            # Add best point marker
            best_iter = self.get_best_iteration(metric_name)
            if best_iter is not None:
                best_value = self.metrics_by_iteration[best_iter][metric_name]
                axes[i].scatter([best_iter], [best_value], color='red', s=100, zorder=5, label='Best')
                axes[i].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
