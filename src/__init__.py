"""
Elite Weld Defect Classifier
Complete pipeline for weld defect detection using modern deep learning
"""

__version__ = "1.0.0"
__author__ = "Claude AI"
__description__ = "State-of-the-art weld defect classification with <3ms inference"

from . import text_removal, synthetic_data, training, export

__all__ = ["text_removal", "synthetic_data", "training", "export"]
