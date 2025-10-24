"""
Base Dataset Loader
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import random


class BaseDatasetLoader(ABC):
    """
    Base class for dataset loaders used in meta-learning experiments
    """

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or Path("./data/datasets")
        self.data_path.mkdir(parents=True, exist_ok=True)
        self._data = None
        self._train_data = None
        self._test_data = None

    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        """Load the dataset from source"""
        pass

    @abstractmethod
    def format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Format a single sample for evaluation"""
        pass

    def get_samples(
        self,
        num_samples: Optional[int] = None,
        split: str = 'test',
        shuffle: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get dataset samples

        Args:
            num_samples: Number of samples to return (None = all)
            split: 'train', 'test', or 'all'
            shuffle: Whether to shuffle samples

        Returns:
            List of formatted samples
        """
        if self._data is None:
            self.load_data()

        # Select split
        if split == 'train':
            data = self._train_data or []
        elif split == 'test':
            data = self._test_data or []
        else:
            data = self._data or []

        # Shuffle if requested
        if shuffle:
            data = random.sample(data, len(data))

        # Limit samples
        if num_samples is not None:
            data = data[:num_samples]

        # Format samples
        return [self.format_sample(sample) for sample in data]

    def split_data(self, train_ratio: float = 0.8):
        """
        Split data into train/test sets

        Args:
            train_ratio: Ratio of data to use for training
        """
        if self._data is None:
            self.load_data()

        random.shuffle(self._data)
        split_idx = int(len(self._data) * train_ratio)

        self._train_data = self._data[:split_idx]
        self._test_data = self._data[split_idx:]

        print(f"Split data: {len(self._train_data)} train, {len(self._test_data)} test")

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if self._data is None:
            self.load_data()

        stats = {
            'total_samples': len(self._data) if self._data else 0,
            'train_samples': len(self._train_data) if self._train_data else 0,
            'test_samples': len(self._test_data) if self._test_data else 0,
        }

        return stats

    def save_cache(self, cache_path: Optional[Path] = None):
        """Save processed data to cache"""
        if cache_path is None:
            cache_path = self.data_path / f"{self.__class__.__name__}_cache.json"

        cache_data = {
            'data': self._data,
            'train_data': self._train_data,
            'test_data': self._test_data
        }

        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)

        print(f"Cache saved to {cache_path}")

    def load_cache(self, cache_path: Optional[Path] = None) -> bool:
        """
        Load data from cache

        Returns:
            True if cache loaded successfully
        """
        if cache_path is None:
            cache_path = self.data_path / f"{self.__class__.__name__}_cache.json"

        if not cache_path.exists():
            return False

        with open(cache_path, 'r') as f:
            cache_data = json.load(f)

        self._data = cache_data.get('data')
        self._train_data = cache_data.get('train_data')
        self._test_data = cache_data.get('test_data')

        print(f"Cache loaded from {cache_path}")
        return True
