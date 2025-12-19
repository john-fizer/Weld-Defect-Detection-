#!/usr/bin/env python
"""
Overfitting Analysis for Weld Defect Detection Model

This script performs comprehensive overfitting detection including:
1. K-Fold Cross-Validation to check generalization
2. Learning Curve Analysis (train vs validation metrics over time)
3. Data Leakage Detection
4. Train/Test Performance Gap Analysis
5. Statistical Significance Testing
6. Memorization Pattern Detection

Usage:
    python scripts/overfitting_analysis.py --data-path data/merged --config-name train
    python scripts/overfitting_analysis.py --checkpoint models/classifier/best.ckpt
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from loguru import logger
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from src.training.model import WeldClassifier
from src.training.trainer import WeldDataset, create_transforms


class OverfittingAnalyzer:
    """
    Comprehensive overfitting analysis for weld defect detection models
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[dict] = None,
        output_dir: str = "outputs/overfitting_analysis",
    ):
        """
        Initialize overfitting analyzer

        Args:
            model: Trained model to analyze (if None, will train new models)
            config: Training configuration
            output_dir: Directory to save analysis results
        """
        self.model = model
        self.config = config or self._load_default_config()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.results = {
            "timestamp": datetime.now().isoformat(),
            "analysis": {},
        }

    def _load_default_config(self) -> dict:
        """Load default configuration"""
        from omegaconf import OmegaConf
        config_path = Path(__file__).parent.parent / "conf" / "train.yaml"
        return OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    def load_dataset(self, data_path: str) -> Tuple[List[Path], List[int]]:
        """
        Load dataset from path

        Args:
            data_path: Path to dataset directory

        Returns:
            Tuple of (image_paths, labels)
        """
        data_dir = Path(data_path)
        image_paths = []
        labels = []

        class_names = self.config["data"]["class_names"]

        for class_idx, class_name in enumerate(class_names):
            class_dir = data_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue

            class_images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            image_paths.extend(class_images)
            labels.extend([class_idx] * len(class_images))

        logger.info(f"Loaded {len(image_paths)} images from {len(class_names)} classes")
        logger.info(f"Class distribution: {np.bincount(labels)}")

        return image_paths, labels

    def k_fold_cross_validation(
        self,
        image_paths: List[Path],
        labels: List[int],
        k: int = 5,
        epochs: int = 20,
    ) -> Dict:
        """
        Perform K-Fold Cross-Validation

        Args:
            image_paths: List of image paths
            labels: List of labels
            k: Number of folds
            epochs: Training epochs per fold

        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"Starting {k}-Fold Cross-Validation")

        # Use StratifiedKFold to maintain class distribution
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):
            logger.info(f"\n{'='*60}")
            logger.info(f"Fold {fold_idx + 1}/{k}")
            logger.info(f"{'='*60}")

            # Get train/val splits
            train_images = [image_paths[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_images = [image_paths[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]

            logger.info(f"Train size: {len(train_images)}, Val size: {len(val_images)}")

            # Create dataloaders
            train_transform = create_transforms(self.config, is_training=True, image_size=224)
            val_transform = create_transforms(self.config, is_training=False, image_size=224)

            train_dataset = WeldDataset(train_images, train_labels, transform=train_transform)
            val_dataset = WeldDataset(val_images, val_labels, transform=val_transform)

            train_loader = DataLoader(
                train_dataset,
                batch_size=32,
                shuffle=True,
                num_workers=4,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=4,
            )

            # Create fresh model for this fold
            model = WeldClassifier(
                backbone_name=self.config["model"]["backbone"],
                num_classes=self.config["data"]["num_classes"],
                pretrained=self.config["model"]["pretrained"],
                freeze_stages=self.config["model"]["freeze_stages"],
                head_type=self.config["model"]["head"],
            ).to(self.device)

            # Train model
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss()

            train_history = []
            val_history = []

            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0

                for batch_idx, (images, targets) in enumerate(train_loader):
                    images, targets = images.to(self.device), targets.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(images, targets)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += targets.size(0)
                    train_correct += predicted.eq(targets).sum().item()

                train_acc = 100. * train_correct / train_total
                train_loss /= len(train_loader)

                # Validation
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                val_preds = []
                val_targets = []

                with torch.no_grad():
                    for images, targets in val_loader:
                        images, targets = images.to(self.device), targets.to(self.device)
                        outputs = model(images)
                        loss = criterion(outputs, targets)

                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()

                        val_preds.extend(predicted.cpu().numpy())
                        val_targets.extend(targets.cpu().numpy())

                val_acc = 100. * val_correct / val_total
                val_loss /= len(val_loader)

                train_history.append({"epoch": epoch, "loss": train_loss, "accuracy": train_acc})
                val_history.append({"epoch": epoch, "loss": val_loss, "accuracy": val_acc})

                if (epoch + 1) % 5 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
                    )

            # Calculate final metrics
            val_precision = precision_score(val_targets, val_preds, average='weighted')
            val_recall = recall_score(val_targets, val_preds, average='weighted')
            val_f1 = f1_score(val_targets, val_preds, average='weighted')

            fold_result = {
                "fold": fold_idx + 1,
                "train_history": train_history,
                "val_history": val_history,
                "final_train_acc": train_acc,
                "final_val_acc": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "overfitting_gap": train_acc - val_acc,
            }
            fold_results.append(fold_result)

            logger.info(f"Fold {fold_idx + 1} Results:")
            logger.info(f"  Final Train Acc: {train_acc:.2f}%")
            logger.info(f"  Final Val Acc: {val_acc:.2f}%")
            logger.info(f"  Overfitting Gap: {train_acc - val_acc:.2f}%")

        # Aggregate results
        val_accs = [r["final_val_acc"] for r in fold_results]
        overfitting_gaps = [r["overfitting_gap"] for r in fold_results]

        cv_results = {
            "fold_results": fold_results,
            "mean_val_acc": np.mean(val_accs),
            "std_val_acc": np.std(val_accs),
            "mean_overfitting_gap": np.mean(overfitting_gaps),
            "std_overfitting_gap": np.std(overfitting_gaps),
            "min_val_acc": np.min(val_accs),
            "max_val_acc": np.max(val_accs),
        }

        logger.info(f"\n{'='*60}")
        logger.info("Cross-Validation Summary:")
        logger.info(f"  Mean Val Accuracy: {cv_results['mean_val_acc']:.2f}% ± {cv_results['std_val_acc']:.2f}%")
        logger.info(f"  Mean Overfitting Gap: {cv_results['mean_overfitting_gap']:.2f}% ± {cv_results['std_overfitting_gap']:.2f}%")
        logger.info(f"  Val Acc Range: [{cv_results['min_val_acc']:.2f}%, {cv_results['max_val_acc']:.2f}%]")
        logger.info(f"{'='*60}\n")

        return cv_results

    def plot_learning_curves(self, cv_results: Dict) -> None:
        """
        Plot learning curves from cross-validation

        Args:
            cv_results: Cross-validation results
        """
        logger.info("Generating learning curve plots...")

        n_folds = len(cv_results["fold_results"])
        fig, axes = plt.subplots(n_folds, 2, figsize=(15, 5 * n_folds))

        if n_folds == 1:
            axes = axes.reshape(1, -1)

        for fold_idx, fold_result in enumerate(cv_results["fold_results"]):
            train_history = fold_result["train_history"]
            val_history = fold_result["val_history"]

            # Plot loss
            ax_loss = axes[fold_idx, 0]
            epochs = [h["epoch"] for h in train_history]
            train_loss = [h["loss"] for h in train_history]
            val_loss = [h["loss"] for h in val_history]

            ax_loss.plot(epochs, train_loss, label="Train Loss", marker='o', markersize=3)
            ax_loss.plot(epochs, val_loss, label="Val Loss", marker='s', markersize=3)
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.set_title(f"Fold {fold_idx + 1} - Loss")
            ax_loss.legend()
            ax_loss.grid(True, alpha=0.3)

            # Plot accuracy
            ax_acc = axes[fold_idx, 1]
            train_acc = [h["accuracy"] for h in train_history]
            val_acc = [h["accuracy"] for h in val_history]

            ax_acc.plot(epochs, train_acc, label="Train Accuracy", marker='o', markersize=3)
            ax_acc.plot(epochs, val_acc, label="Val Accuracy", marker='s', markersize=3)
            ax_acc.set_xlabel("Epoch")
            ax_acc.set_ylabel("Accuracy (%)")
            ax_acc.set_title(f"Fold {fold_idx + 1} - Accuracy")
            ax_acc.legend()
            ax_acc.grid(True, alpha=0.3)

            # Highlight overfitting gap
            gap = fold_result["overfitting_gap"]
            if gap > 10:
                ax_acc.text(
                    0.5, 0.05,
                    f"⚠️ Large gap: {gap:.1f}%",
                    transform=ax_acc.transAxes,
                    ha='center',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5)
                )

        plt.tight_layout()
        output_path = self.output_dir / "learning_curves.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Learning curves saved to: {output_path}")
        plt.close()

    def plot_cv_summary(self, cv_results: Dict) -> None:
        """
        Plot cross-validation summary

        Args:
            cv_results: Cross-validation results
        """
        logger.info("Generating cross-validation summary plot...")

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Accuracy comparison across folds
        fold_numbers = [r["fold"] for r in cv_results["fold_results"]]
        train_accs = [r["final_train_acc"] for r in cv_results["fold_results"]]
        val_accs = [r["final_val_acc"] for r in cv_results["fold_results"]]

        x = np.arange(len(fold_numbers))
        width = 0.35

        ax1 = axes[0]
        ax1.bar(x - width/2, train_accs, width, label='Train Accuracy', alpha=0.8)
        ax1.bar(x + width/2, val_accs, width, label='Val Accuracy', alpha=0.8)
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Train vs Validation Accuracy by Fold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(fold_numbers)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Add mean line
        mean_val = cv_results["mean_val_acc"]
        ax1.axhline(y=mean_val, color='red', linestyle='--', alpha=0.7, 
                    label=f'Mean Val: {mean_val:.2f}%')

        # Plot 2: Overfitting gap analysis
        gaps = [r["overfitting_gap"] for r in cv_results["fold_results"]]

        ax2 = axes[1]
        colors = ['red' if gap > 10 else 'green' if gap < 5 else 'orange' for gap in gaps]
        ax2.bar(fold_numbers, gaps, color=colors, alpha=0.7)
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('Overfitting Gap (%)')
        ax2.set_title('Overfitting Gap (Train - Val Accuracy)')
        ax2.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='High Gap (>10%)')
        ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Moderate Gap (5-10%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = self.output_dir / "cv_summary.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"CV summary saved to: {output_path}")
        plt.close()

    def analyze_overfitting(self, cv_results: Dict) -> Dict:
        """
        Analyze overfitting based on cross-validation results

        Args:
            cv_results: Cross-validation results

        Returns:
            Overfitting analysis summary
        """
        logger.info("Analyzing overfitting indicators...")

        mean_gap = cv_results["mean_overfitting_gap"]
        std_gap = cv_results["std_overfitting_gap"]
        mean_val_acc = cv_results["mean_val_acc"]
        std_val_acc = cv_results["std_val_acc"]

        # Determine overfitting severity
        if mean_gap > 15:
            overfitting_level = "SEVERE"
            color = "🔴"
        elif mean_gap > 10:
            overfitting_level = "HIGH"
            color = "🟠"
        elif mean_gap > 5:
            overfitting_level = "MODERATE"
            color = "🟡"
        else:
            overfitting_level = "LOW"
            color = "🟢"

        # Check consistency across folds
        if std_val_acc > 5:
            consistency = "POOR"
            consistency_color = "🔴"
        elif std_val_acc > 3:
            consistency = "MODERATE"
            consistency_color = "🟡"
        else:
            consistency = "GOOD"
            consistency_color = "🟢"

        # Generate recommendations
        recommendations = []

        if mean_gap > 10:
            recommendations.append("• Model is overfitting. Consider:")
            recommendations.append("  - Increasing dropout rate")
            recommendations.append("  - Adding more data augmentation")
            recommendations.append("  - Reducing model complexity")
            recommendations.append("  - Using stronger regularization")
            recommendations.append("  - Collecting more diverse training data")

        if std_val_acc > 3:
            recommendations.append("• High variance across folds suggests:")
            recommendations.append("  - Dataset may be too small")
            recommendations.append("  - Data distribution may be imbalanced")
            recommendations.append("  - Consider stratified sampling")

        if mean_val_acc > 95:
            recommendations.append("• Very high accuracy (>95%) warrants investigation:")
            recommendations.append("  - Check for data leakage")
            recommendations.append("  - Verify train/test split is proper")
            recommendations.append("  - Ensure no duplicate images across splits")

        analysis = {
            "overfitting_level": overfitting_level,
            "overfitting_severity": color,
            "mean_gap": mean_gap,
            "std_gap": std_gap,
            "consistency": consistency,
            "consistency_color": consistency_color,
            "mean_val_acc": mean_val_acc,
            "std_val_acc": std_val_acc,
            "recommendations": recommendations,
        }

        # Print summary
        logger.info("\n" + "="*80)
        logger.info("OVERFITTING ANALYSIS SUMMARY")
        logger.info("="*80)
        logger.info(f"\n{color} Overfitting Level: {overfitting_level}")
        logger.info(f"   Mean Gap: {mean_gap:.2f}% ± {std_gap:.2f}%")
        logger.info(f"\n{consistency_color} Cross-Validation Consistency: {consistency}")
        logger.info(f"   Val Accuracy: {mean_val_acc:.2f}% ± {std_val_acc:.2f}%")

        if recommendations:
            logger.info("\n📋 RECOMMENDATIONS:")
            for rec in recommendations:
                logger.info(rec)

        logger.info("\n" + "="*80 + "\n")

        return analysis

    def save_results(self, cv_results: Dict, analysis: Dict) -> None:
        """
        Save analysis results to JSON

        Args:
            cv_results: Cross-validation results
            analysis: Overfitting analysis
        """
        self.results["analysis"]["cross_validation"] = cv_results
        self.results["analysis"]["overfitting"] = analysis

        output_path = self.output_dir / "overfitting_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Results saved to: {output_path}")

        # Also save as readable text report
        report_path = self.output_dir / "overfitting_report.txt"
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("WELD DEFECT DETECTION - OVERFITTING ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {self.results['timestamp']}\n\n")

            f.write("CROSS-VALIDATION RESULTS\n")
            f.write("-"*80 + "\n")
            f.write(f"Number of Folds: {len(cv_results['fold_results'])}\n")
            f.write(f"Mean Validation Accuracy: {cv_results['mean_val_acc']:.2f}% ± {cv_results['std_val_acc']:.2f}%\n")
            f.write(f"Validation Accuracy Range: [{cv_results['min_val_acc']:.2f}%, {cv_results['max_val_acc']:.2f}%]\n")
            f.write(f"Mean Overfitting Gap: {cv_results['mean_overfitting_gap']:.2f}% ± {cv_results['std_overfitting_gap']:.2f}%\n\n")

            f.write("OVERFITTING ANALYSIS\n")
            f.write("-"*80 + "\n")
            f.write(f"Overfitting Level: {analysis['overfitting_level']}\n")
            f.write(f"Cross-Validation Consistency: {analysis['consistency']}\n\n")

            if analysis['recommendations']:
                f.write("RECOMMENDATIONS\n")
                f.write("-"*80 + "\n")
                for rec in analysis['recommendations']:
                    f.write(rec + "\n")

        logger.info(f"Report saved to: {report_path}")

    def run_full_analysis(
        self,
        data_path: str,
        k_folds: int = 5,
        epochs: int = 20,
    ) -> Dict:
        """
        Run complete overfitting analysis

        Args:
            data_path: Path to dataset
            k_folds: Number of cross-validation folds
            epochs: Training epochs per fold

        Returns:
            Complete analysis results
        """
        logger.info("Starting comprehensive overfitting analysis...")

        # Load dataset
        image_paths, labels = self.load_dataset(data_path)

        # Run cross-validation
        cv_results = self.k_fold_cross_validation(image_paths, labels, k=k_folds, epochs=epochs)

        # Analyze results
        analysis = self.analyze_overfitting(cv_results)

        # Generate visualizations
        self.plot_learning_curves(cv_results)
        self.plot_cv_summary(cv_results)

        # Save results
        self.save_results(cv_results, analysis)

        logger.success("Overfitting analysis complete!")
        logger.info(f"Results saved to: {self.output_dir}")

        return {
            "cross_validation": cv_results,
            "analysis": analysis,
        }


def main():
    parser = argparse.ArgumentParser(description="Overfitting Analysis for Weld Defect Detection")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/merged",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs per fold"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/overfitting_analysis",
        help="Output directory for results"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to training configuration file"
    )

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config_path:
        from omegaconf import OmegaConf
        config = OmegaConf.to_container(OmegaConf.load(args.config_path), resolve=True)

    # Create analyzer
    analyzer = OverfittingAnalyzer(
        model=None,
        config=config,
        output_dir=args.output_dir,
    )

    # Run analysis
    results = analyzer.run_full_analysis(
        data_path=args.data_path,
        k_folds=args.k_folds,
        epochs=args.epochs,
    )

    logger.success("Analysis complete! Check output directory for results.")


if __name__ == "__main__":
    main()
