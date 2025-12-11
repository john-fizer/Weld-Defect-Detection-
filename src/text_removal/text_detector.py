"""
Text Detection Module - EasyOCR Integration
Zero-shot handwritten text detection on faint marker ink
"""

import numpy as np
import cv2
import easyocr
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class TextDetection:
    """Container for detected text region"""
    bbox: np.ndarray           # Shape: (4, 2) - four corner points
    text: str                  # Recognized text
    confidence: float          # Detection confidence
    mask: Optional[np.ndarray] = None  # Binary mask of text region


class TextDetector:
    """
    EasyOCR-based text detector optimized for industrial weld images
    Detects both printed labels and handwritten marker annotations
    """

    def __init__(
        self,
        languages: List[str] = ['en'],
        gpu: bool = True,
        min_size: int = 10,
        text_threshold: float = 0.7,
        low_text: float = 0.4,
        link_threshold: float = 0.4,
        canvas_size: int = 2560,
        mag_ratio: float = 1.0,
    ):
        """
        Initialize EasyOCR text detector

        Args:
            languages: List of language codes (e.g., ['en', 'ch_sim'])
            gpu: Use GPU acceleration
            min_size: Minimum text size in pixels
            text_threshold: Text detection confidence threshold
            low_text: Low text score threshold for CRAFT
            link_threshold: Link between characters threshold
            canvas_size: Maximum image size for processing
            mag_ratio: Magnification ratio for detection
        """
        logger.info(f"Initializing EasyOCR with languages: {languages}")

        self.reader = easyocr.Reader(
            languages,
            gpu=gpu,
            detector=True,
            recognizer=True,
            verbose=False,
            quantize=False,  # Don't quantize for best accuracy
            cudnn_benchmark=gpu,
        )

        self.min_size = min_size
        self.text_threshold = text_threshold
        self.low_text = low_text
        self.link_threshold = link_threshold
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio

        logger.success("EasyOCR initialized successfully")

    def detect(
        self,
        image: np.ndarray,
        paragraph: bool = False,
    ) -> List[TextDetection]:
        """
        Detect text in image

        Args:
            image: Input image (RGB or BGR)
            paragraph: Group text into paragraphs

        Returns:
            List of TextDetection objects
        """
        # Ensure image is RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3:
            # Assume BGR, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run EasyOCR detection
        try:
            results = self.reader.readtext(
                image,
                detail=1,  # Return bounding boxes
                paragraph=paragraph,
                min_size=self.min_size,
                text_threshold=self.text_threshold,
                low_text=self.low_text,
                link_threshold=self.link_threshold,
                canvas_size=self.canvas_size,
                mag_ratio=self.mag_ratio,
                slope_ths=0.1,
                ycenter_ths=0.5,
                height_ths=0.5,
                width_ths=0.5,
                add_margin=0.1,
            )
        except Exception as e:
            logger.error(f"EasyOCR detection failed: {e}")
            return []

        # Convert to TextDetection objects
        detections = []
        for bbox, text, confidence in results:
            bbox_array = np.array(bbox, dtype=np.float32)

            detection = TextDetection(
                bbox=bbox_array,
                text=text,
                confidence=confidence,
            )
            detections.append(detection)

        logger.info(f"Detected {len(detections)} text regions")
        return detections

    def create_masks(
        self,
        image: np.ndarray,
        detections: List[TextDetection],
        dilation_kernel_size: int = 30,
        dilation_iterations: int = 2,
        blur_kernel_size: int = 5,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Create dilated masks for text regions (prevents inpainting halos)

        Args:
            image: Original image
            detections: List of text detections
            dilation_kernel_size: Kernel size for mask dilation (20-30px)
            dilation_iterations: Number of dilation iterations
            blur_kernel_size: Kernel size for mask edge smoothing

        Returns:
            combined_mask: Single mask with all text regions (H, W)
            individual_masks: List of individual text masks
        """
        h, w = image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        individual_masks = []

        for detection in detections:
            # Create mask for this detection
            mask = np.zeros((h, w), dtype=np.uint8)

            # Fill polygon
            bbox_int = detection.bbox.astype(np.int32)
            cv2.fillPoly(mask, [bbox_int], 255)

            # Dilate mask to cover surrounding area (prevents halos)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (dilation_kernel_size, dilation_kernel_size)
            )
            mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)

            # Smooth edges with Gaussian blur
            if blur_kernel_size > 0:
                mask = cv2.GaussianBlur(mask, (blur_kernel_size, blur_kernel_size), 0)

            # Add to combined mask
            combined_mask = cv2.bitwise_or(combined_mask, mask)
            individual_masks.append(mask)

            # Store mask in detection object
            detection.mask = mask

        logger.info(f"Created {len(individual_masks)} dilated masks")
        return combined_mask, individual_masks

    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[TextDetection],
        show_text: bool = True,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """
        Visualize detected text regions on image

        Args:
            image: Original image
            detections: List of text detections
            show_text: Draw detected text
            show_confidence: Draw confidence scores

        Returns:
            Visualization image
        """
        vis = image.copy()

        for detection in detections:
            # Draw bounding box
            bbox = detection.bbox.astype(np.int32)
            cv2.polylines(vis, [bbox], True, (0, 255, 0), 2)

            # Draw text and confidence
            if show_text or show_confidence:
                label_parts = []
                if show_text:
                    label_parts.append(detection.text)
                if show_confidence:
                    label_parts.append(f"{detection.confidence:.2f}")

                label = " - ".join(label_parts)

                # Put text above bounding box
                text_pos = (bbox[0][0], bbox[0][1] - 10)
                cv2.putText(
                    vis, label, text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

        return vis

    def process_image(
        self,
        image_path: Path,
        output_mask_path: Optional[Path] = None,
        output_vis_path: Optional[Path] = None,
        **mask_kwargs,
    ) -> Tuple[List[TextDetection], np.ndarray]:
        """
        End-to-end processing: detect text and create mask

        Args:
            image_path: Path to input image
            output_mask_path: Path to save combined mask
            output_vis_path: Path to save visualization
            mask_kwargs: Additional arguments for create_masks()

        Returns:
            detections: List of text detections
            combined_mask: Combined mask of all text regions
        """
        logger.info(f"Processing image: {image_path}")

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Detect text
        detections = self.detect(image)

        # Create masks
        combined_mask, _ = self.create_masks(image, detections, **mask_kwargs)

        # Save mask if requested
        if output_mask_path:
            output_mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_mask_path), combined_mask)
            logger.info(f"Saved mask to: {output_mask_path}")

        # Save visualization if requested
        if output_vis_path:
            vis = self.visualize_detections(image, detections)
            output_vis_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_vis_path), vis)
            logger.info(f"Saved visualization to: {output_vis_path}")

        return detections, combined_mask


def batch_process_images(
    input_dir: Path,
    output_mask_dir: Path,
    output_vis_dir: Optional[Path] = None,
    detector: Optional[TextDetector] = None,
    file_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp'],
    **detector_kwargs,
) -> None:
    """
    Batch process all images in a directory

    Args:
        input_dir: Directory containing input images
        output_mask_dir: Directory to save masks
        output_vis_dir: Directory to save visualizations (optional)
        detector: Pre-initialized TextDetector (creates new if None)
        file_extensions: List of file extensions to process
        detector_kwargs: Arguments for TextDetector initialization
    """
    input_dir = Path(input_dir)
    output_mask_dir = Path(output_mask_dir)

    if detector is None:
        detector = TextDetector(**detector_kwargs)

    # Find all images
    image_paths = []
    for ext in file_extensions:
        image_paths.extend(input_dir.glob(f"**/*{ext}"))
        image_paths.extend(input_dir.glob(f"**/*{ext.upper()}"))

    logger.info(f"Found {len(image_paths)} images to process")

    # Process each image
    for image_path in image_paths:
        try:
            # Create output paths
            relative_path = image_path.relative_to(input_dir)
            mask_path = output_mask_dir / relative_path.with_suffix('.png')
            vis_path = output_vis_dir / relative_path if output_vis_dir else None

            # Process
            detector.process_image(
                image_path,
                output_mask_path=mask_path,
                output_vis_path=vis_path,
            )

        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            continue

    logger.success(f"Batch processing complete: {len(image_paths)} images")


if __name__ == "__main__":
    # Example usage
    detector = TextDetector(
        languages=['en'],
        gpu=True,
        min_size=10,
        text_threshold=0.7,
    )

    # Process single image
    detections, mask = detector.process_image(
        Path("data/raw/sample_weld.jpg"),
        output_mask_path=Path("data/masks/sample_weld_mask.png"),
        output_vis_path=Path("data/visualizations/sample_weld_vis.jpg"),
        dilation_kernel_size=30,
        dilation_iterations=2,
    )

    print(f"Found {len(detections)} text regions")
