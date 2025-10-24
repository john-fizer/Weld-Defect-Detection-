"""
CommonsenseQA Dataset Loader

Dataset: https://www.tau-nlp.org/commonsenseqa
Multiple-choice questions requiring commonsense reasoning
"""

import json
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path
from .base import BaseDatasetLoader


class CommonsenseQALoader(BaseDatasetLoader):
    """
    Loader for CommonsenseQA dataset

    Question format:
    - Question text
    - 5 multiple choice options (A-E)
    - Correct answer key
    - Concept set (optional)
    """

    DATASET_URL = "https://s3.amazonaws.com/commonsenseqa/train_rand_split.jsonl"
    DATASET_URL_DEV = "https://s3.amazonaws.com/commonsenseqa/dev_rand_split.jsonl"

    def __init__(self, data_path: Optional[Path] = None, use_dev: bool = False):
        super().__init__(data_path)
        self.use_dev = use_dev
        self.dataset_file = "commonsense_qa_dev.jsonl" if use_dev else "commonsense_qa_train.jsonl"

    def download_data(self) -> Path:
        """Download dataset if not already present"""
        file_path = self.data_path / self.dataset_file

        if file_path.exists():
            print(f"Dataset already exists at {file_path}")
            return file_path

        print(f"Downloading CommonsenseQA dataset...")
        url = self.DATASET_URL_DEV if self.use_dev else self.DATASET_URL

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with open(file_path, 'wb') as f:
                f.write(response.content)

            print(f"Dataset downloaded to {file_path}")
            return file_path

        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Creating sample dataset for demonstration...")
            return self._create_sample_dataset()

    def _create_sample_dataset(self) -> Path:
        """Create a sample dataset for demonstration purposes"""
        file_path = self.data_path / self.dataset_file

        sample_data = [
            {
                "id": "sample_1",
                "question": {
                    "stem": "The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?",
                    "choices": [
                        {"label": "A", "text": "ignore"},
                        {"label": "B", "text": "enforce"},
                        {"label": "C", "text": "authorise"},
                        {"label": "D", "text": "revert"},
                        {"label": "E", "text": "disregard"}
                    ]
                },
                "answerKey": "A"
            },
            {
                "id": "sample_2",
                "question": {
                    "stem": "Where would you find a snake in a tropical forest?",
                    "choices": [
                        {"label": "A", "text": "tree"},
                        {"label": "B", "text": "pet shop"},
                        {"label": "C", "text": "georgia"},
                        {"label": "D", "text": "garden"},
                        {"label": "E", "text": "outdoors"}
                    ]
                },
                "answerKey": "A"
            },
            {
                "id": "sample_3",
                "question": {
                    "stem": "What do people do when they are agreeing with each other?",
                    "choices": [
                        {"label": "A", "text": "smiling"},
                        {"label": "B", "text": "nodding"},
                        {"label": "C", "text": "make cakes"},
                        {"label": "D", "text": "trade places"},
                        {"label": "E", "text": "farting"}
                    ]
                },
                "answerKey": "B"
            },
            {
                "id": "sample_4",
                "question": {
                    "stem": "Where would you put uncooked crab meat?",
                    "choices": [
                        {"label": "A", "text": "wharf"},
                        {"label": "B", "text": "red lobster"},
                        {"label": "C", "text": "tidepools"},
                        {"label": "D", "text": "boss's office"},
                        {"label": "E", "text": "refrigerator"}
                    ]
                },
                "answerKey": "E"
            },
            {
                "id": "sample_5",
                "question": {
                    "stem": "The man was going fishing instead of work, what is he seeking?",
                    "choices": [
                        {"label": "A", "text": "relaxation"},
                        {"label": "B", "text": "nice car"},
                        {"label": "C", "text": "tropical fish"},
                        {"label": "D", "text": "adventure"},
                        {"label": "E", "text": "boredom"}
                    ]
                },
                "answerKey": "A"
            }
        ]

        with open(file_path, 'w') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')

        print(f"Sample dataset created at {file_path}")
        return file_path

    def load_data(self) -> List[Dict[str, Any]]:
        """Load CommonsenseQA data"""
        # Try to load from cache first
        if self.load_cache():
            return self._data

        # Download if needed
        data_file = self.download_data()

        # Parse JSONL file
        data = []
        with open(data_file, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        self._data = data

        # Auto-split
        self.split_data()

        # Cache for next time
        self.save_cache()

        print(f"Loaded {len(data)} CommonsenseQA samples")
        return data

    def format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format CommonsenseQA sample for evaluation

        Returns:
            {
                'input': formatted question with choices,
                'expected': correct answer,
                'answer_key': letter of correct answer,
                'metadata': additional info
            }
        """
        question_data = sample['question']
        stem = question_data['stem']
        choices = question_data['choices']

        # Format choices
        choices_text = "\n".join([
            f"{choice['label']}. {choice['text']}"
            for choice in choices
        ])

        # Create formatted input
        formatted_input = f"Question: {stem}\n\nChoices:\n{choices_text}\n\nAnswer:"

        # Get correct answer
        answer_key = sample['answerKey']
        correct_choice = next(
            (c for c in choices if c['label'] == answer_key),
            None
        )

        return {
            'id': sample.get('id', 'unknown'),
            'input': formatted_input,
            'expected': correct_choice['text'] if correct_choice else '',
            'answer_key': answer_key,
            'question': stem,
            'choices': {c['label']: c['text'] for c in choices},
            'metadata': {
                'concept_set': sample.get('question_concept', ''),
                'dataset': 'commonsense_qa'
            }
        }

    def create_prompt_template(self, include_examples: bool = True) -> str:
        """
        Create a base prompt template for CommonsenseQA

        Args:
            include_examples: Whether to include few-shot examples

        Returns:
            Prompt template string
        """
        base_prompt = """Answer the following multiple-choice question using your commonsense reasoning.

{examples}

{question}

Provide only the letter of the correct answer (A, B, C, D, or E)."""

        if include_examples:
            examples = """Example 1:
Question: Where would you put uncooked crab meat?
Choices:
A. wharf
B. red lobster
C. tidepools
D. boss's office
E. refrigerator
Answer: E

Example 2:
Question: What do people do when they are agreeing with each other?
Choices:
A. smiling
B. nodding
C. make cakes
D. trade places
E. farting
Answer: B

"""
            base_prompt = base_prompt.format(examples=examples, question="{question}")
        else:
            base_prompt = base_prompt.format(examples="", question="{question}")

        return base_prompt

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        base_stats = super().get_statistics()

        if self._data:
            # Count answer distribution
            answer_dist = {}
            for sample in self._data:
                key = sample.get('answerKey', 'unknown')
                answer_dist[key] = answer_dist.get(key, 0) + 1

            base_stats['answer_distribution'] = answer_dist

        return base_stats
