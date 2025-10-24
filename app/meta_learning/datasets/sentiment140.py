"""
Sentiment140 Dataset Loader

Dataset: http://help.sentiment140.com/for-students
Twitter sentiment classification (positive/negative)
"""

import csv
import requests
import zipfile
from typing import List, Dict, Any, Optional
from pathlib import Path
from .base import BaseDatasetLoader


class Sentiment140Loader(BaseDatasetLoader):
    """
    Loader for Sentiment140 dataset

    Format:
    - Binary sentiment classification (0=negative, 4=positive)
    - Tweet text
    - User info
    - Timestamp
    """

    DATASET_URL = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"

    # For demo, we'll create sample data
    def __init__(self, data_path: Optional[Path] = None):
        super().__init__(data_path)
        self.dataset_file = "sentiment140.csv"

    def download_data(self) -> Path:
        """Download dataset if not already present"""
        file_path = self.data_path / self.dataset_file

        if file_path.exists():
            print(f"Dataset already exists at {file_path}")
            return file_path

        print(f"Creating sample Sentiment140 dataset...")
        return self._create_sample_dataset()

    def _create_sample_dataset(self) -> Path:
        """Create a sample dataset for demonstration purposes"""
        file_path = self.data_path / self.dataset_file

        # Sample tweets with sentiment labels
        sample_data = [
            # Negative (0)
            [0, "1467810369", "Mon Apr 06 22:19:45 PDT 2009", "NO_QUERY", "user1", "is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!"],
            [0, "1467810672", "Mon Apr 06 22:19:49 PDT 2009", "NO_QUERY", "user2", "I was sick all night. Still feeling terrible. This is not good."],
            [0, "1467810917", "Mon Apr 06 22:19:53 PDT 2009", "NO_QUERY", "user3", "My boss is a jerk. Having the worst day at work ever."],
            [0, "1467811184", "Mon Apr 06 22:19:57 PDT 2009", "NO_QUERY", "user4", "Can't believe I failed that test. So disappointed in myself."],
            [0, "1467811193", "Mon Apr 06 22:19:57 PDT 2009", "NO_QUERY", "user5", "Traffic is terrible today. Going to be late again."],

            # Positive (4)
            [4, "1467811372", "Mon Apr 06 22:20:00 PDT 2009", "NO_QUERY", "user6", "Just got promoted! Best day ever! So excited!"],
            [4, "1467811592", "Mon Apr 06 22:20:03 PDT 2009", "NO_QUERY", "user7", "Beautiful weather today. Perfect for a picnic in the park with friends!"],
            [4, "1467811795", "Mon Apr 06 22:20:05 PDT 2009", "NO_QUERY", "user8", "Finally finished that project! Feels amazing. Time to celebrate!"],
            [4, "1467812025", "Mon Apr 06 22:20:09 PDT 2009", "NO_QUERY", "user9", "Got accepted into my dream school! I'm so happy I could cry!"],
            [4, "1467812133", "Mon Apr 06 22:20:10 PDT 2009", "NO_QUERY", "user10", "This concert is incredible! Having the time of my life!"],

            # More negatives
            [0, "1467812278", "Mon Apr 06 22:20:13 PDT 2009", "NO_QUERY", "user11", "My computer just crashed and I lost all my work. Why does this always happen to me?"],
            [0, "1467812416", "Mon Apr 06 22:20:15 PDT 2009", "NO_QUERY", "user12", "Broke up with my girlfriend today. Feeling awful."],

            # More positives
            [4, "1467812579", "Mon Apr 06 22:20:17 PDT 2009", "NO_QUERY", "user13", "Just landed my dream job! Can't believe it! So grateful!"],
            [4, "1467812697", "Mon Apr 06 22:20:19 PDT 2009", "NO_QUERY", "user14", "Love my new apartment! Best decision ever to move here."],
            [4, "1467812820", "Mon Apr 06 22:20:21 PDT 2009", "NO_QUERY", "user15", "Coffee with best friends. These are the moments I live for."],
        ]

        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(sample_data)

        print(f"Sample dataset created at {file_path}")
        return file_path

    def load_data(self) -> List[Dict[str, Any]]:
        """Load Sentiment140 data"""
        # Try to load from cache first
        if self.load_cache():
            return self._data

        # Download/create if needed
        data_file = self.download_data()

        # Parse CSV file
        data = []
        with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 6:
                    data.append({
                        'polarity': int(row[0]),  # 0=negative, 4=positive
                        'id': row[1],
                        'date': row[2],
                        'query': row[3],
                        'user': row[4],
                        'text': row[5]
                    })

        self._data = data

        # Auto-split
        self.split_data()

        # Cache for next time
        self.save_cache()

        print(f"Loaded {len(data)} Sentiment140 samples")
        return data

    def format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format Sentiment140 sample for evaluation

        Returns:
            {
                'input': tweet text,
                'expected': sentiment label (positive/negative),
                'label': numeric label,
                'metadata': additional info
            }
        """
        # Convert polarity to readable label
        polarity = sample['polarity']
        if polarity == 0:
            sentiment = 'negative'
        elif polarity == 4:
            sentiment = 'positive'
        else:
            sentiment = 'neutral'

        return {
            'id': sample['id'],
            'input': sample['text'],
            'expected': sentiment,
            'label': polarity,
            'text': sample['text'],
            'metadata': {
                'user': sample['user'],
                'date': sample['date'],
                'dataset': 'sentiment140'
            }
        }

    def create_prompt_template(self, include_examples: bool = True) -> str:
        """
        Create a base prompt template for Sentiment140

        Args:
            include_examples: Whether to include few-shot examples

        Returns:
            Prompt template string
        """
        base_prompt = """Classify the sentiment of the following tweet as either "positive" or "negative".

{examples}

Tweet: {tweet}

Sentiment:"""

        if include_examples:
            examples = """Examples:

Tweet: "Just got promoted! Best day ever! So excited!"
Sentiment: positive

Tweet: "I was sick all night. Still feeling terrible. This is not good."
Sentiment: negative

Tweet: "Beautiful weather today. Perfect for a picnic in the park with friends!"
Sentiment: positive

"""
            base_prompt = base_prompt.format(examples=examples, tweet="{tweet}")
        else:
            base_prompt = base_prompt.format(examples="", tweet="{tweet}")

        return base_prompt

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        base_stats = super().get_statistics()

        if self._data:
            # Count sentiment distribution
            sentiment_dist = {'positive': 0, 'negative': 0, 'neutral': 0}

            for sample in self._data:
                polarity = sample.get('polarity', -1)
                if polarity == 0:
                    sentiment_dist['negative'] += 1
                elif polarity == 4:
                    sentiment_dist['positive'] += 1
                else:
                    sentiment_dist['neutral'] += 1

            base_stats['sentiment_distribution'] = sentiment_dist

            # Calculate balance
            total = sum(sentiment_dist.values())
            if total > 0:
                base_stats['balance'] = {
                    k: v/total for k, v in sentiment_dist.items()
                }

        return base_stats

    def balance_dataset(self, max_per_class: Optional[int] = None):
        """
        Balance the dataset to have equal samples per class

        Args:
            max_per_class: Maximum samples per class
        """
        if self._data is None:
            self.load_data()

        # Group by sentiment
        positive = [s for s in self._data if s['polarity'] == 4]
        negative = [s for s in self._data if s['polarity'] == 0]

        # Find minimum or use max_per_class
        min_count = min(len(positive), len(negative))
        if max_per_class is not None:
            min_count = min(min_count, max_per_class)

        # Sample equally
        import random
        balanced_data = (
            random.sample(positive, min_count) +
            random.sample(negative, min_count)
        )

        random.shuffle(balanced_data)
        self._data = balanced_data

        print(f"Balanced dataset: {min_count} positive, {min_count} negative")

        # Re-split
        self.split_data()
