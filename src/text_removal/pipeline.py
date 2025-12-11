"""
End-to-End Text Removal Pipeline
Combines EasyOCR detection with LaMa inpainting for 100% text-free images
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List
from loguru import logger
from tqdm import tqdm

from .text_detector import TextDetector
from .lama_inpainter import LamaInpainter, SimpleLamaInpainter


class TextRemovalPipeline:
    """
    Complete pipeline for text detection and removal
    Stage 1 of Elite Weld Defect Classifier workflow
    """

    def __init__(
        self,
        detector_config: Optional[dict] = None,
        inpainter_config: Optional[dict] = None,
        use_lama: bool = True,
    ):
        """
        Initialize text removal pipeline

        Args:
            detector_config: Configuration for TextDetector
            inpainter_config: Configuration for LamaInpainter
            use_lama: Use LaMa (True) or OpenCV fallback (False)
        """
        logger.info("Initializing Text Removal Pipeline")

        # Default configurations
        detector_config = detector_config or {
            "languages": ['en'],
            "gpu": True,
            "min_size": 10,
            "text_threshold": 0.7,
            "low_text": 0.4,
            "link_threshold": 0.4,
        }

        inpainter_config = inpainter_config or {
            "checkpoint": "big-lama",
            "device": "cuda",
            "batch_size": 4,
        }

        # Initialize detector
        self.detector = TextDetector(**detector_config)

        # Initialize inpainter
        if use_lama:
            try:
                self.inpainter = LamaInpainter(**inpainter_config)
                logger.success("Using LaMa inpainting")
            except ImportError:
                logger.warning("LaMa not available, falling back to OpenCV")
                self.inpainter = SimpleLamaInpainter()
        else:
            self.inpainter = SimpleLamaInpainter()
            logger.info("Using OpenCV inpainting (fallback)")

        self.mask_config = {
            "dilation_kernel_size": 30,
            "dilation_iterations": 2,
            "blur_kernel_size": 5,
        }

    def process_single_image(
        self,
        image_path: Path,
        output_path: Path,
        save_mask: bool = False,
        save_visualization: bool = False,
    ) -> dict:
        """
        Process a single image through the complete pipeline

        Args:
            image_path: Input image path
            output_path: Output inpainted image path
            save_mask: Save intermediate mask
            save_visualization: Save detection visualization

        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Processing: {image_path.name}")

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Step 1: Detect text
        detections = self.detector.detect(image)

        stats = {
            "image_path": str(image_path),
            "num_detections": len(detections),
            "has_text": len(detections) > 0,
        }

        # If no text detected, just copy the image
        if len(detections) == 0:
            logger.info(f"No text detected in {image_path.name}, copying original")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), image)
            stats["processing"] = "copy"
            return stats

        # Step 2: Create dilated masks (prevents halos)
        combined_mask, individual_masks = self.detector.create_masks(
            image,
            detections,
            **self.mask_config
        )

        # Save mask if requested
        if save_mask:
            mask_path = output_path.parent / "masks" / output_path.name
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), combined_mask)
            stats["mask_path"] = str(mask_path)

        # Save visualization if requested
        if save_visualization:
            vis = self.detector.visualize_detections(image, detections)
            vis_path = output_path.parent / "visualizations" / output_path.name
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vis_path), vis)
            stats["visualization_path"] = str(vis_path)

        # Step 3: Inpaint text regions
        if isinstance(self.inpainter, LamaInpainter):
            result = self.inpainter.inpaint(image, combined_mask, return_time=True)
            inpainted = result.inpainted_image
            stats["inference_time"] = result.inference_time
        else:
            inpainted = self.inpainter.inpaint(image, combined_mask)
            stats["inference_time"] = None

        # Step 4: Save result
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), inpainted)
        stats["output_path"] = str(output_path)
        stats["processing"] = "inpainted"

        logger.success(f"Completed: {image_path.name} ({len(detections)} text regions removed)")

        return stats

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        file_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp'],
        save_masks: bool = False,
        save_visualizations: bool = False,
        recursive: bool = True,
    ) -> List[dict]:
        """
        Process all images in a directory

        Args:
            input_dir: Input directory
            output_dir: Output directory
            file_extensions: File extensions to process
            save_masks: Save intermediate masks
            save_visualizations: Save detection visualizations
            recursive: Process subdirectories recursively

        Returns:
            List of processing statistics for each image
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # Find all images
        image_paths = []
        for ext in file_extensions:
            pattern = f"**/*{ext}" if recursive else f"*{ext}"
            image_paths.extend(input_dir.glob(pattern))
            image_paths.extend(input_dir.glob(pattern.upper()))

        # Remove duplicates and sort
        image_paths = sorted(set(image_paths))

        logger.info(f"Found {len(image_paths)} images to process")

        # Process each image with progress bar
        all_stats = []
        for image_path in tqdm(image_paths, desc="Removing text"):
            try:
                # Maintain directory structure
                relative_path = image_path.relative_to(input_dir)
                output_path = output_dir / relative_path

                stats = self.process_single_image(
                    image_path,
                    output_path,
                    save_mask=save_masks,
                    save_visualization=save_visualizations,
                )
                all_stats.append(stats)

            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                all_stats.append({
                    "image_path": str(image_path),
                    "processing": "failed",
                    "error": str(e),
                })
                continue

        # Summary statistics
        total_processed = len(all_stats)
        total_with_text = sum(1 for s in all_stats if s.get("has_text", False))
        total_failed = sum(1 for s in all_stats if s.get("processing") == "failed")

        logger.success(f"\n{'='*60}")
        logger.success(f"Text Removal Pipeline Complete")
        logger.success(f"{'='*60}")
        logger.success(f"Total images: {total_processed}")
        logger.success(f"Images with text: {total_with_text}")
        logger.success(f"Images failed: {total_failed}")
        logger.success(f"Output directory: {output_dir}")
        logger.success(f"{'='*60}\n")

        return all_stats


def run_text_removal_from_config(config: dict) -> None:
    """
    Run text removal pipeline from Hydra config

    Args:
        config: Hydra configuration dictionary
    """
    # Extract text removal config
    text_config = config.get("text_removal", {})

    if not text_config.get("enabled", True):
        logger.info("Text removal disabled in config, skipping")
        return

    # Extract paths
    raw_path = Path(config["data"]["raw_path"])
    clean_path = Path(config["data"]["clean_real_path"])

    # Initialize pipeline
    pipeline = TextRemovalPipeline(
        detector_config={
            "languages": text_config["easyocr"]["languages"],
            "gpu": text_config["easyocr"]["gpu"],
            "min_size": text_config["easyocr"]["min_size"],
            "text_threshold": text_config["easyocr"]["text_threshold"],
            "low_text": text_config["easyocr"]["low_text"],
            "link_threshold": text_config["easyocr"]["link_threshold"],
        },
        inpainter_config={
            "model_path": Path(text_config["lama"]["model_path"]),
            "checkpoint": text_config["lama"]["checkpoint"],
            "device": text_config["lama"]["device"],
            "batch_size": text_config["lama"]["batch_size"],
        },
        use_lama=True,
    )

    # Update mask config
    pipeline.mask_config = {
        "dilation_kernel_size": text_config["mask"]["dilation_kernel_size"],
        "dilation_iterations": text_config["mask"]["dilation_iterations"],
        "blur_kernel_size": text_config["mask"]["blur_kernel_size"],
    }

    # Process directory
    pipeline.process_directory(
        input_dir=raw_path,
        output_dir=clean_path,
        save_masks=True,
        save_visualizations=True,
    )


if __name__ == "__main__":
    # Example usage
    pipeline = TextRemovalPipeline(
        detector_config={
            "languages": ['en'],
            "gpu": True,
            "text_threshold": 0.7,
        },
        inpainter_config={
            "checkpoint": "big-lama",
            "device": "cuda",
        },
    )

    # Process directory
    stats = pipeline.process_directory(
        input_dir=Path("data/raw"),
        output_dir=Path("data/clean_real"),
        save_masks=True,
        save_visualizations=True,
    )

    # Print summary
    print(f"\nProcessed {len(stats)} images")
    print(f"Images with text: {sum(1 for s in stats if s.get('has_text'))}")
