#!/usr/bin/env python
"""
Quick Overfitting Check for Weld Defect Detection Model

Fast validation to check if a model is overfitting:
- Single train/val split test
- Quick metrics comparison
- Simple pass/fail indicators

Usage:
    python scripts/quick_overfitting_check.py --data-path data/merged
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from loguru import logger
from typing import Tuple, List
import random

from src.training.model import WeldClassifier
from src.training.trainer import WeldDataset, create_transforms


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(data_path: str, train_ratio: float = 0.8) -> Tuple:
    """Load and split data"""
    data_dir = Path(data_path)
    image_paths = []
    labels = []

    class_names = ["good_weld", "bad_weld"]

    for class_idx, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            logger.warning(f"Class directory not found: {class_dir}")
            continue

        class_images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        image_paths.extend(class_images)
        labels.extend([class_idx] * len(class_images))

    # Shuffle
    indices = list(range(len(image_paths)))
    random.shuffle(indices)
    image_paths = [image_paths[i] for i in indices]
    labels = [labels[i] for i in indices]

    # Split
    split_idx = int(len(image_paths) * train_ratio)
    train_images = image_paths[:split_idx]
    train_labels = labels[:split_idx]
    val_images = image_paths[split_idx:]
    val_labels = labels[split_idx:]

    logger.info(f"Total images: {len(image_paths)}")
    logger.info(f"Train: {len(train_images)}, Val: {len(val_images)}")
    logger.info(f"Class distribution - Train: {np.bincount(train_labels)}, Val: {np.bincount(val_labels)}")

    return train_images, train_labels, val_images, val_labels


def evaluate(model, dataloader, device):
    """Evaluate model on dataset"""
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_preds) * 100
    precision = precision_score(all_targets, all_preds, average='weighted') * 100
    recall = recall_score(all_targets, all_preds, average='weighted') * 100
    f1 = f1_score(all_targets, all_preds, average='weighted') * 100
    avg_loss = total_loss / len(dataloader)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "loss": avg_loss,
        "predictions": all_preds,
        "targets": all_targets,
    }


def quick_overfitting_check(
    data_path: str,
    epochs: int = 15,
    learning_rate: float = 1e-4,
) -> dict:
    """
    Quick overfitting check

    Args:
        data_path: Path to dataset
        epochs: Number of training epochs
        learning_rate: Learning rate

    Returns:
        Dictionary with results
    """
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    train_images, train_labels, val_images, val_labels = load_data(data_path)

    # Check if we have enough data
    if len(train_images) < 20:
        logger.error("Insufficient training data! Need at least 20 images.")
        return None

    # Create config
    config = {
        "augmentation": {
            "train": {
                "horizontal_flip": 0.5,
                "vertical_flip": 0.3,
                "rotate_limit": 15,
                "brightness_limit": 0.2,
                "contrast_limit": 0.2,
                "hue_shift_limit": 10,
                "sat_shift_limit": 20,
                "val_shift_limit": 20,
                "blur_limit": 3,
                "gaussian_noise": 0.02,
                "shift_limit": 0.1,
                "scale_limit": 0.2,
            }
        }
    }

    # Create datasets
    train_transform = create_transforms(config, is_training=True, image_size=224)
    val_transform = create_transforms(config, is_training=False, image_size=224)

    train_dataset = WeldDataset(train_images, train_labels, transform=train_transform)
    val_dataset = WeldDataset(val_images, val_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    # Create model
    logger.info("Creating model...")
    model = WeldClassifier(
        backbone_name="convnextv2_nano.fcmae_ft_in22k_in1k_384",
        num_classes=2,
        pretrained=True,
        freeze_stages=2,
        head_type="arcface",
    ).to(device)

    # Optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    logger.info("Starting training...")
    train_history = []
    val_history = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

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
        val_results = evaluate(model, val_loader, device)
        val_acc = val_results["accuracy"]
        val_loss = val_results["loss"]

        train_history.append({"epoch": epoch + 1, "loss": train_loss, "accuracy": train_acc})
        val_history.append({"epoch": epoch + 1, "loss": val_loss, "accuracy": val_acc})

        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

    # Final evaluation
    logger.info("\nFinal Evaluation:")
    train_results = evaluate(model, train_loader, device)
    val_results = evaluate(model, val_loader, device)

    logger.info(f"Train Accuracy: {train_results['accuracy']:.2f}%")
    logger.info(f"Val Accuracy: {val_results['accuracy']:.2f}%")

    # Calculate overfitting metrics
    overfitting_gap = train_results["accuracy"] - val_results["accuracy"]

    # Generate report
    logger.info("\n" + "="*80)
    logger.info("OVERFITTING CHECK RESULTS")
    logger.info("="*80)
    logger.info(f"\n📊 Final Metrics:")
    logger.info(f"   Train Accuracy: {train_results['accuracy']:.2f}%")
    logger.info(f"   Val Accuracy:   {val_results['accuracy']:.2f}%")
    logger.info(f"   Gap:            {overfitting_gap:.2f}%")
    logger.info(f"\n   Val Precision:  {val_results['precision']:.2f}%")
    logger.info(f"   Val Recall:     {val_results['recall']:.2f}%")
    logger.info(f"   Val F1:         {val_results['f1']:.2f}%")

    # Overfitting assessment
    if overfitting_gap > 15:
        status = "🔴 SEVERE OVERFITTING"
        verdict = "FAIL"
    elif overfitting_gap > 10:
        status = "🟠 HIGH OVERFITTING"
        verdict = "CAUTION"
    elif overfitting_gap > 5:
        status = "🟡 MODERATE OVERFITTING"
        verdict = "PASS (with concerns)"
    else:
        status = "🟢 LOW OVERFITTING"
        verdict = "PASS"

    logger.info(f"\n{status}")
    logger.info(f"Verdict: {verdict}")

    # Recommendations
    logger.info("\n📋 Recommendations:")
    if overfitting_gap > 10:
        logger.info("   • Model shows significant overfitting")
        logger.info("   • Increase regularization (dropout, weight decay)")
        logger.info("   • Add more data augmentation")
        logger.info("   • Consider reducing model complexity")
        logger.info("   • Collect more diverse training data")
    elif overfitting_gap > 5:
        logger.info("   • Moderate overfitting detected")
        logger.info("   • Monitor validation performance closely")
        logger.info("   • Consider early stopping")
    else:
        logger.info("   • Model generalizes well")
        logger.info("   • Continue with current training setup")

    # Warning about claimed 97% accuracy
    if val_results["accuracy"] < 85:
        logger.info("\n⚠️  WARNING:")
        logger.info("   • Validation accuracy is significantly lower than claimed 97%")
        logger.info("   • Claimed accuracy may be based on training set")
        logger.info("   • Recommend full cross-validation analysis")

    logger.info("\n" + "="*80 + "\n")

    return {
        "train_history": train_history,
        "val_history": val_history,
        "final_train_metrics": train_results,
        "final_val_metrics": val_results,
        "overfitting_gap": overfitting_gap,
        "status": status,
        "verdict": verdict,
    }


def main():
    parser = argparse.ArgumentParser(description="Quick Overfitting Check")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/merged",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )

    args = parser.parse_args()

    # Run check
    results = quick_overfitting_check(
        data_path=args.data_path,
        epochs=args.epochs,
        learning_rate=args.lr,
    )

    if results:
        logger.success("Quick overfitting check complete!")
    else:
        logger.error("Quick overfitting check failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
