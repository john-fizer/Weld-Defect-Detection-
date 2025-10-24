"""
Experiment Logger

Comprehensive logging for meta-learning experiments
"""

import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime


class ExperimentLogger:
    """
    Structured logging for experiments

    Logs:
    - Experiment configurations
    - Performance metrics
    - System events
    - Errors and warnings
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        experiment_name: str = "experiment"
    ):
        self.log_dir = log_dir or Path("./data/meta_learning/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Text log
        log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"

        # Configure logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # JSON log for structured data
        self.json_log_file = self.log_dir / f"{experiment_name}_{timestamp}.jsonl"
        self.json_log = []

        self.logger.info(f"Experiment logger initialized: {experiment_name}")

    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration"""
        self.logger.info("="*70)
        self.logger.info("EXPERIMENT CONFIGURATION")
        self.logger.info("="*70)

        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")

        self._append_json_log({
            'type': 'config',
            'timestamp': datetime.now().isoformat(),
            'config': config
        })

    def log_iteration(
        self,
        iteration: int,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log iteration results"""
        self.logger.info(f"\nIteration {iteration} Results:")
        self.logger.info("-"*40)

        for metric_name, value in metrics.items():
            self.logger.info(f"  {metric_name}: {value:.4f}")

        self._append_json_log({
            'type': 'iteration',
            'timestamp': datetime.now().isoformat(),
            'iteration': iteration,
            'metrics': metrics,
            'metadata': metadata or {}
        })

    def log_event(self, event_type: str, message: str, data: Optional[Dict] = None):
        """Log a system event"""
        self.logger.info(f"[{event_type}] {message}")

        self._append_json_log({
            'type': 'event',
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'message': message,
            'data': data or {}
        })

    def log_error(self, error: Exception, context: str = ""):
        """Log an error"""
        self.logger.error(f"Error in {context}: {str(error)}", exc_info=True)

        self._append_json_log({
            'type': 'error',
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'error': str(error),
            'error_type': type(error).__name__
        })

    def log_warning(self, message: str, data: Optional[Dict] = None):
        """Log a warning"""
        self.logger.warning(message)

        self._append_json_log({
            'type': 'warning',
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'data': data or {}
        })

    def log_comparison(self, comparison_results: Dict[str, Any]):
        """Log experiment comparison results"""
        self.logger.info("\n" + "="*70)
        self.logger.info("COMPARISON RESULTS")
        self.logger.info("="*70)

        if 'meta_prompting_wins' in comparison_results:
            winner = "Meta-Prompting" if comparison_results['meta_prompting_wins'] else "Baseline"
            self.logger.info(f"\nWinner: {winner}")
            self.logger.info(f"Margin: {comparison_results.get('win_margin', 0):.4f}")

        self._append_json_log({
            'type': 'comparison',
            'timestamp': datetime.now().isoformat(),
            'results': comparison_results
        })

    def log_summary(self, summary: Dict[str, Any]):
        """Log experiment summary"""
        self.logger.info("\n" + "="*70)
        self.logger.info("EXPERIMENT SUMMARY")
        self.logger.info("="*70)

        for key, value in summary.items():
            if isinstance(value, dict):
                self.logger.info(f"\n{key}:")
                for sub_key, sub_value in value.items():
                    self.logger.info(f"  {sub_key}: {sub_value}")
            else:
                self.logger.info(f"{key}: {value}")

        self._append_json_log({
            'type': 'summary',
            'timestamp': datetime.now().isoformat(),
            'summary': summary
        })

    def _append_json_log(self, entry: Dict[str, Any]):
        """Append entry to JSON log"""
        self.json_log.append(entry)

        # Also write to file immediately
        with open(self.json_log_file, 'a') as f:
            f.write(json.dumps(entry, default=str) + '\n')

    def close(self):
        """Close logger"""
        self.logger.info("\n" + "="*70)
        self.logger.info("Experiment logging complete")
        self.logger.info("="*70)

        # Close handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
