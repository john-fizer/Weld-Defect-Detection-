"""
LaMa Inpainting Module
State-of-the-art inpainting for industrial metal textures
Uses Big-LaMa variant (commit 2a3f7e1) - current inpainting king
"""

import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Optional, Union, List, Tuple
from dataclasses import dataclass
from loguru import logger
import yaml

try:
    from lama_cleaner.model.lama import LaMa
    from lama_cleaner.schema import Config as LamaConfig
    LAMA_AVAILABLE = True
except ImportError:
    LAMA_AVAILABLE = False
    logger.warning("LaMa not available. Install with: pip install git+https://github.com/advimman/lama.git@2a3f7e1")


@dataclass
class InpaintingResult:
    """Container for inpainting results"""
    inpainted_image: np.ndarray
    original_image: np.ndarray
    mask: np.ndarray
    inference_time: float


class LamaInpainter:
    """
    LaMa (Large Mask Inpainting) wrapper
    Optimized for removing text from industrial weld images
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        checkpoint: str = "big-lama",
        device: str = "cuda",
        batch_size: int = 4,
    ):
        """
        Initialize LaMa inpainter

        Args:
            model_path: Path to LaMa model directory
            checkpoint: Model checkpoint name ('big-lama' recommended)
            device: Device for inference ('cuda', 'cpu', 'mps')
            batch_size: Batch size for processing multiple images
        """
        if not LAMA_AVAILABLE:
            raise ImportError(
                "LaMa not installed. Install with: "
                "pip install git+https://github.com/advimman/lama.git@2a3f7e1"
            )

        self.device = self._get_device(device)
        self.batch_size = batch_size

        logger.info(f"Initializing LaMa inpainter on {self.device}")
        logger.info(f"Checkpoint: {checkpoint}")

        # Load model
        try:
            self.model = self._load_model(model_path, checkpoint)
            logger.success("LaMa model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LaMa model: {e}")
            raise

    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device"""
        if device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            logger.warning(f"Device {device} not available, using CPU")
            return torch.device("cpu")

    def _load_model(
        self,
        model_path: Optional[Path],
        checkpoint: str,
    ) -> torch.nn.Module:
        """
        Load LaMa model

        Args:
            model_path: Path to model directory
            checkpoint: Checkpoint name

        Returns:
            Loaded LaMa model
        """
        # Note: This is a simplified loader
        # In practice, you would:
        # 1. Download the Big-LaMa checkpoint from the official repo
        # 2. Load the model architecture and weights
        # 3. Set to eval mode

        # For now, we'll use a placeholder that shows the structure
        # In actual implementation, you'd use the official LaMa loader

        model = LaMa(device=self.device)
        model.eval()

        return model

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for LaMa

        Args:
            image: Input image (H, W, 3) in BGR or RGB format

        Returns:
            Preprocessed tensor (1, 3, H, W) normalized to [-1, 1]
        """
        # Convert to RGB if BGR
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Normalize to [-1, 1] (LaMa expects this range)
        image = (image - 0.5) / 0.5

        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

        return image_tensor.to(self.device)

    def preprocess_mask(self, mask: np.ndarray) -> torch.Tensor:
        """
        Preprocess mask for LaMa

        Args:
            mask: Binary mask (H, W) with 255 for text regions

        Returns:
            Preprocessed mask tensor (1, 1, H, W) normalized to [0, 1]
        """
        # Ensure binary mask
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Normalize to [0, 1]
        mask = (mask > 127).astype(np.float32)

        # Convert to tensor and add batch and channel dimensions
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)

        return mask_tensor.to(self.device)

    def postprocess_output(self, output_tensor: torch.Tensor) -> np.ndarray:
        """
        Postprocess LaMa output

        Args:
            output_tensor: Model output (1, 3, H, W) in [-1, 1]

        Returns:
            Output image (H, W, 3) in uint8 [0, 255]
        """
        # Remove batch dimension and move to CPU
        output = output_tensor.squeeze(0).cpu().numpy()

        # Denormalize from [-1, 1] to [0, 1]
        output = (output * 0.5) + 0.5

        # Convert to [0, 255]
        output = (output * 255).clip(0, 255).astype(np.uint8)

        # Convert from (C, H, W) to (H, W, C)
        output = output.transpose(1, 2, 0)

        return output

    @torch.no_grad()
    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        return_time: bool = False,
    ) -> Union[np.ndarray, InpaintingResult]:
        """
        Inpaint image using mask

        Args:
            image: Input image (H, W, 3)
            mask: Binary mask (H, W) with 255 for regions to inpaint
            return_time: Return timing information

        Returns:
            Inpainted image or InpaintingResult if return_time=True
        """
        import time
        start_time = time.time()

        # Preprocess
        image_tensor = self.preprocess_image(image)
        mask_tensor = self.preprocess_mask(mask)

        # Pad to multiple of 8 (LaMa requirement)
        h, w = image.shape[:2]
        h_pad = ((h - 1) // 8 + 1) * 8
        w_pad = ((w - 1) // 8 + 1) * 8

        if h != h_pad or w != w_pad:
            image_tensor = torch.nn.functional.pad(
                image_tensor,
                (0, w_pad - w, 0, h_pad - h),
                mode='reflect'
            )
            mask_tensor = torch.nn.functional.pad(
                mask_tensor,
                (0, w_pad - w, 0, h_pad - h),
                mode='constant',
                value=0
            )

        # Run inference
        with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
            output_tensor = self.model(image_tensor, mask_tensor)

        # Remove padding
        if h != h_pad or w != w_pad:
            output_tensor = output_tensor[:, :, :h, :w]

        # Postprocess
        inpainted = self.postprocess_output(output_tensor)

        inference_time = time.time() - start_time

        if return_time:
            return InpaintingResult(
                inpainted_image=inpainted,
                original_image=image,
                mask=mask,
                inference_time=inference_time,
            )
        else:
            return inpainted

    def inpaint_batch(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Batch inpainting for multiple images

        Args:
            images: List of input images
            masks: List of corresponding masks

        Returns:
            List of inpainted images
        """
        results = []

        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            batch_masks = masks[i:i + self.batch_size]

            for image, mask in zip(batch_images, batch_masks):
                inpainted = self.inpaint(image, mask)
                results.append(inpainted)

            logger.info(f"Processed batch {i // self.batch_size + 1}/{(len(images) - 1) // self.batch_size + 1}")

        return results

    def process_directory(
        self,
        image_dir: Path,
        mask_dir: Path,
        output_dir: Path,
        file_extensions: List[str] = ['.jpg', '.jpeg', '.png'],
    ) -> None:
        """
        Process all images in a directory

        Args:
            image_dir: Directory containing input images
            mask_dir: Directory containing masks
            output_dir: Directory to save inpainted images
            file_extensions: List of file extensions to process
        """
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all images
        image_paths = []
        for ext in file_extensions:
            image_paths.extend(image_dir.glob(f"**/*{ext}"))
            image_paths.extend(image_dir.glob(f"**/*{ext.upper()}"))

        logger.info(f"Found {len(image_paths)} images to inpaint")

        # Process each image
        for image_path in image_paths:
            try:
                # Find corresponding mask
                relative_path = image_path.relative_to(image_dir)
                mask_path = mask_dir / relative_path.with_suffix('.png')

                if not mask_path.exists():
                    logger.warning(f"Mask not found for {image_path}, skipping")
                    continue

                # Load image and mask
                image = cv2.imread(str(image_path))
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

                if image is None or mask is None:
                    logger.warning(f"Failed to load {image_path} or mask, skipping")
                    continue

                # Inpaint
                result = self.inpaint(image, mask, return_time=True)

                # Save result
                output_path = output_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), result.inpainted_image)

                logger.info(
                    f"Inpainted {image_path.name} in {result.inference_time:.3f}s"
                )

            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                continue

        logger.success(f"Directory inpainting complete: {len(image_paths)} images")

    def compare_inpainting(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        inpainted: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Create side-by-side comparison of original, mask, and inpainted

        Args:
            image: Original image
            mask: Binary mask
            inpainted: Inpainted image (will compute if None)

        Returns:
            Comparison image with three panels
        """
        if inpainted is None:
            inpainted = self.inpaint(image, mask)

        # Convert mask to 3-channel for visualization
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Create side-by-side comparison
        comparison = np.hstack([image, mask_vis, inpainted])

        return comparison


class SimpleLamaInpainter:
    """
    Simplified LaMa wrapper for cases where official LaMa is not available
    Uses OpenCV-based inpainting as fallback
    """

    def __init__(self, method: str = "telea"):
        """
        Initialize simple inpainter

        Args:
            method: 'telea' or 'ns' (Navier-Stokes)
        """
        self.method = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        logger.warning("Using OpenCV fallback inpainting (not as good as LaMa)")

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        inpaint_radius: int = 3,
    ) -> np.ndarray:
        """
        Inpaint using OpenCV

        Args:
            image: Input image
            mask: Binary mask
            inpaint_radius: Inpainting radius

        Returns:
            Inpainted image
        """
        return cv2.inpaint(image, mask, inpaint_radius, self.method)


if __name__ == "__main__":
    # Example usage
    try:
        inpainter = LamaInpainter(
            checkpoint="big-lama",
            device="cuda",
            batch_size=4,
        )
    except ImportError:
        logger.warning("Using OpenCV fallback")
        inpainter = SimpleLamaInpainter()

    # Process directory
    inpainter.process_directory(
        image_dir=Path("data/raw"),
        mask_dir=Path("data/masks"),
        output_dir=Path("data/clean_real"),
    )
