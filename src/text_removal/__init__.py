"""
Text Removal Pipeline for Weld Images
Combines EasyOCR detection with LaMa inpainting
"""

from .text_detector import TextDetector, TextDetection, batch_process_images
from .lama_inpainter import LamaInpainter, SimpleLamaInpainter, InpaintingResult

__all__ = [
    "TextDetector",
    "TextDetection",
    "LamaInpainter",
    "SimpleLamaInpainter",
    "InpaintingResult",
    "batch_process_images",
]
