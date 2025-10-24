"""
Visualization Tools for Meta-Learning Experiments
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import json


class MetaLearningVisualizer:
    """
    Visualization tools for meta-learning experiments

    Generates:
    - Performance trends
    - Comparison charts
    - Drift/alignment/coherence plots
    - Strategy effectiveness analysis
    """

    def __init__(self, save_path: Optional[Path] = None):
        self.save_path = save_path or Path("./data/meta_learning/visualizations")
        self.save_path.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

    def plot_curriculum_learning(
        self,
        performance_history: List[Dict[str, float]],
        title: str = "Adaptive Curriculum Learning Progress",
        save_name: Optional[str] = None
    ):
        """
        Plot curriculum learning progress

        Args:
            performance_history: List of performance metrics per iteration
            title: Plot title
            save_name: Filename to save (optional)
        """
        iterations = list(range(1, len(performance_history) + 1))

        # Extract metrics
        metrics_to_plot = ['accuracy', 'f1', 'precision', 'recall']
        available_metrics = {
            metric: [h.get(metric, 0) for h in performance_history]
            for metric in metrics_to_plot
            if any(metric in h for h in performance_history)
        }

        if not available_metrics:
            print("No metrics to plot")
            return

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        for metric_name, values in available_metrics.items():
            ax.plot(iterations, values, marker='o', label=metric_name.capitalize(), linewidth=2)

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Highlight best iteration
        if 'accuracy' in available_metrics:
            best_iter = np.argmax(available_metrics['accuracy'])
            best_value = available_metrics['accuracy'][best_iter]
            ax.scatter([iterations[best_iter]], [best_value],
                      color='red', s=200, zorder=5, marker='*',
                      label=f'Best ({best_value:.4f})')
            ax.legend(fontsize=10)

        plt.tight_layout()

        if save_name:
            save_path = self.save_path / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_comparison(
        self,
        meta_prompting_history: List[Dict[str, float]],
        baseline_history: List[Dict[str, float]],
        metric: str = 'accuracy',
        title: str = "Meta-Prompting vs Static Baseline",
        save_name: Optional[str] = None
    ):
        """
        Plot comparison between meta-prompting and baseline

        Args:
            meta_prompting_history: Meta-prompting performance history
            baseline_history: Baseline performance history
            metric: Metric to plot
            title: Plot title
            save_name: Filename to save (optional)
        """
        iterations = list(range(1, max(len(meta_prompting_history), len(baseline_history)) + 1))

        meta_values = [h.get(metric, 0) for h in meta_prompting_history]
        baseline_values = [h.get(metric, 0) for h in baseline_history]

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(iterations[:len(meta_values)], meta_values,
                marker='o', label='Meta-Prompting', linewidth=2, color='#2E86AB')
        ax.plot(iterations[:len(baseline_values)], baseline_values,
                marker='s', label='Static Baseline', linewidth=2, color='#A23B72')

        # Shade improvement area
        if len(meta_values) == len(baseline_values):
            ax.fill_between(iterations[:len(meta_values)],
                           meta_values, baseline_values,
                           where=np.array(meta_values) > np.array(baseline_values),
                           alpha=0.3, color='green', label='Meta-Prompting Advantage')

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            save_path = self.save_path / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_clrs_metrics(
        self,
        drift_scores: List[float],
        alignment_scores: List[float],
        coherence_scores: List[float],
        title: str = "CLRS Metrics Over Time",
        save_name: Optional[str] = None
    ):
        """
        Plot CLRS drift, alignment, and coherence metrics

        Args:
            drift_scores: Drift scores per cycle
            alignment_scores: Alignment scores per cycle
            coherence_scores: Coherence scores per cycle
            title: Plot title
            save_name: Filename to save (optional)
        """
        cycles = list(range(1, len(drift_scores) + 1))

        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Drift
        axes[0].plot(cycles, drift_scores, marker='o', color='#E63946', linewidth=2)
        axes[0].axhline(y=0.3, color='orange', linestyle='--', label='Threshold')
        axes[0].set_ylabel('Drift Score', fontsize=11)
        axes[0].set_title('Distribution Drift', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Alignment
        axes[1].plot(cycles, alignment_scores, marker='o', color='#2A9D8F', linewidth=2)
        axes[1].axhline(y=0.7, color='orange', linestyle='--', label='Threshold')
        axes[1].set_ylabel('Alignment Score', fontsize=11)
        axes[1].set_title('User Preference Alignment', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Coherence
        axes[2].plot(cycles, coherence_scores, marker='o', color='#264653', linewidth=2)
        axes[2].axhline(y=0.5, color='orange', linestyle='--', label='Threshold')
        axes[2].set_xlabel('Training Cycle', fontsize=11)
        axes[2].set_ylabel('Coherence Score', fontsize=11)
        axes[2].set_title('Output Coherence', fontsize=12, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()

        if save_name:
            save_path = self.save_path / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_strategy_effectiveness(
        self,
        strategy_analysis: Dict[str, Any],
        title: str = "Evolution Strategy Effectiveness",
        save_name: Optional[str] = None
    ):
        """
        Plot effectiveness of different evolution strategies

        Args:
            strategy_analysis: Dictionary of strategy statistics
            title: Plot title
            save_name: Filename to save (optional)
        """
        strategies = list(strategy_analysis.keys())
        avg_improvements = [stats['avg_improvement'] for stats in strategy_analysis.values()]
        success_rates = [stats['success_rate'] for stats in strategy_analysis.values()]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Average improvement
        colors = plt.cm.viridis(np.linspace(0, 1, len(strategies)))
        bars1 = ax1.bar(strategies, avg_improvements, color=colors)
        ax1.set_xlabel('Strategy', fontsize=11)
        ax1.set_ylabel('Average Improvement', fontsize=11)
        ax1.set_title('Average Improvement per Strategy', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')

        # Success rate
        bars2 = ax2.bar(strategies, success_rates, color=colors)
        ax2.set_xlabel('Strategy', fontsize=11)
        ax2.set_ylabel('Success Rate', fontsize=11)
        ax2.set_title('Success Rate per Strategy', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_name:
            save_path = self.save_path / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def create_dashboard(
        self,
        experiment_results: Dict[str, Any],
        save_name: str = "dashboard.png"
    ):
        """
        Create comprehensive dashboard for experiment results

        Args:
            experiment_results: Complete experiment results
            save_name: Filename to save
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Performance trend
        ax1 = fig.add_subplot(gs[0, :])
        if 'performance_history' in experiment_results:
            history = experiment_results['performance_history']
            iterations = list(range(1, len(history) + 1))
            accuracies = [h.get('accuracy', 0) for h in history]
            ax1.plot(iterations, accuracies, marker='o', linewidth=2, color='#2E86AB')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Performance Over Time', fontweight='bold')
            ax1.grid(True, alpha=0.3)

        # 2. Strategy effectiveness (if available)
        ax2 = fig.add_subplot(gs[1, 0])
        if 'strategy_analysis' in experiment_results:
            strategies = list(experiment_results['strategy_analysis'].keys())
            improvements = [s['avg_improvement'] for s in experiment_results['strategy_analysis'].values()]
            ax2.barh(strategies, improvements, color='#A23B72')
            ax2.set_xlabel('Avg Improvement')
            ax2.set_title('Strategy Effectiveness', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')

        # 3. Metrics summary
        ax3 = fig.add_subplot(gs[1, 1])
        if 'final_performance' in experiment_results:
            metrics = experiment_results['final_performance']
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            ax3.bar(metric_names, metric_values, color='#2A9D8F')
            ax3.set_ylabel('Score')
            ax3.set_title('Final Metrics', fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3, axis='y')

        # 4. Improvement
        ax4 = fig.add_subplot(gs[2, :])
        if 'performance_history' in experiment_results:
            history = experiment_results['performance_history']
            iterations = list(range(1, len(history)))
            improvements = [
                history[i].get('accuracy', 0) - history[i-1].get('accuracy', 0)
                for i in range(1, len(history))
            ]
            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            ax4.bar(iterations, improvements, color=colors, alpha=0.7)
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Improvement')
            ax4.set_title('Per-Iteration Improvement', fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')

        fig.suptitle('Meta-Learning Experiment Dashboard', fontsize=16, fontweight='bold')

        save_path = self.save_path / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dashboard saved: {save_path}")

        plt.close()

    def export_visualization_data(
        self,
        data: Dict[str, Any],
        filename: str = "viz_data.json"
    ):
        """
        Export data for external visualization tools

        Args:
            data: Data to export
            filename: Output filename
        """
        output_path = self.save_path / filename

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Visualization data exported: {output_path}")
