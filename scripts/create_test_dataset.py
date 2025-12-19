#!/usr/bin/env python
"""
Create a small test dataset for validating overfitting analysis tools

Creates synthetic test images to verify the overfitting analysis pipeline works
"""

import numpy as np
from PIL import Image
from pathlib import Path
import random


def create_synthetic_weld_image(size=(224, 224), is_good=True):
    """
    Create a synthetic weld image for testing

    Args:
        size: Image size (width, height)
        is_good: True for good weld, False for bad weld
    """
    # Create base metallic texture
    img = np.random.randint(80, 150, (size[1], size[0], 3), dtype=np.uint8)

    # Add noise
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    if is_good:
        # Good weld: smooth horizontal line pattern
        for i in range(0, size[1], 10):
            img[i:i+3, :] = np.clip(img[i:i+3, :] + 30, 0, 255)
    else:
        # Bad weld: random defects (darker spots)
        num_defects = random.randint(3, 8)
        for _ in range(num_defects):
            x = random.randint(0, size[0] - 30)
            y = random.randint(0, size[1] - 30)
            w = random.randint(10, 30)
            h = random.randint(10, 30)
            img[y:y+h, x:x+w] = np.clip(img[y:y+h, x:x+w] - 60, 0, 255)

    return Image.fromarray(img)


def create_test_dataset(output_dir: str, num_per_class: int = 25):
    """
    Create a test dataset

    Args:
        output_dir: Output directory
        num_per_class: Number of images per class
    """
    output_path = Path(output_dir)

    # Create directories
    good_dir = output_path / "good_weld"
    bad_dir = output_path / "bad_weld"
    good_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating test dataset with {num_per_class} images per class...")

    # Create good weld images
    for i in range(num_per_class):
        img = create_synthetic_weld_image(is_good=True)
        img.save(good_dir / f"good_weld_{i:03d}.jpg", quality=95)

    # Create bad weld images
    for i in range(num_per_class):
        img = create_synthetic_weld_image(is_good=False)
        img.save(bad_dir / f"bad_weld_{i:03d}.jpg", quality=95)

    print(f"✓ Created {num_per_class * 2} test images in {output_path}")
    print(f"  - Good welds: {good_dir}")
    print(f"  - Bad welds: {bad_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create test dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/test_dataset",
        help="Output directory"
    )
    parser.add_argument(
        "--num-per-class",
        type=int,
        default=25,
        help="Number of images per class"
    )

    args = parser.parse_args()

    create_test_dataset(args.output_dir, args.num_per_class)
