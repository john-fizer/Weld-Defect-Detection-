"""
Modular ML training for each prediction technique
Allows independent training, validation, and refinement
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path
from datetime import datetime
from app.config import settings


class TechniqueTrainer:
    """
    Train and evaluate ML models for specific prediction techniques
    Each technique gets its own model
    """

    def __init__(self, technique: str):
        """
        Initialize trainer for a specific technique

        Args:
            technique: Technique name (progressions, transits, zr, lb)
        """
        self.technique = technique
        self.model = None
        self.model_version = None

        self.model_path = Path(settings.ml_model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)

    def train_event_prediction_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        model_type: str = 'random_forest'
    ) -> Dict:
        """
        Train model to predict whether events will occur

        Args:
            X: Feature matrix
            y: Target labels (event occurred or not)
            test_size: Proportion of test set
            model_type: 'random_forest' or 'gradient_boosting'

        Returns:
            Training metrics
        """
        if len(X) < 10:
            return {
                'status': 'insufficient_data',
                'sample_count': len(X),
            }

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )

        # Initialize model
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5 if len(X) >= 50 else 3)

        # Predictions
        y_pred = model.predict(X_test)

        # Save model
        self.model = model
        self.model_version = datetime.now().strftime('%Y%m%d_%H%M%S')

        metrics = {
            'technique': self.technique,
            'model_type': model_type,
            'model_version': self.model_version,
            'sample_count': len(X),
            'train_score': train_score,
            'test_score': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'status': 'trained',
        }

        return metrics

    def save_model(self, model_name: Optional[str] = None):
        """Save trained model to disk"""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        if model_name is None:
            model_name = f"{self.technique}_{self.model_version}.joblib"

        model_file = self.model_path / model_name
        joblib.dump(self.model, model_file)

        print(f"Model saved to {model_file}")
        return model_file

    def load_model(self, model_name: str):
        """Load a trained model from disk"""
        model_file = self.model_path / model_name

        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")

        self.model = joblib.load(model_file)
        self.model_version = model_name

        print(f"Model loaded from {model_file}")

    def predict_event_probability(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of event occurrence

        Args:
            X: Feature matrix

        Returns:
            Probability array
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")

        probabilities = self.model.predict_proba(X)[:, 1]  # Probability of positive class
        return probabilities

    def evaluate_on_new_data(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate model on new feedback data

        Args:
            X: New features
            y: New labels

        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model loaded.")

        score = self.model.score(X, y)
        y_pred = self.model.predict(X)

        return {
            'technique': self.technique,
            'model_version': self.model_version,
            'accuracy': score,
            'classification_report': classification_report(y, y_pred, output_dict=True),
            'sample_count': len(X),
        }

    def incremental_training(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray
    ) -> Dict:
        """
        Incrementally update model with new feedback

        Args:
            X_new: New features
            y_new: New labels

        Returns:
            Updated metrics
        """
        if self.model is None:
            raise ValueError("No model loaded. Train initial model first.")

        # For tree-based models, we need to retrain
        # Store old model for comparison
        old_version = self.model_version

        # Retrain with new data added
        # In production, you'd combine old and new data
        self.model.fit(X_new, y_new)

        # Update version
        self.model_version = datetime.now().strftime('%Y%m%d_%H%M%S')

        return {
            'technique': self.technique,
            'old_version': old_version,
            'new_version': self.model_version,
            'new_samples_added': len(X_new),
            'status': 'updated',
        }


class ModelRegistry:
    """
    Manage multiple model versions for A/B testing and rollback
    """

    def __init__(self):
        self.registry_path = Path(settings.ml_model_path) / 'registry.json'
        self.models = self._load_registry()

    def _load_registry(self) -> Dict:
        """Load model registry from disk"""
        if self.registry_path.exists():
            import json
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_registry(self):
        """Save model registry to disk"""
        import json
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.models, f, indent=2, default=str)

    def register_model(
        self,
        technique: str,
        model_version: str,
        metrics: Dict,
        is_production: bool = False
    ):
        """
        Register a trained model

        Args:
            technique: Technique name
            model_version: Version identifier
            metrics: Training metrics
            is_production: Whether this is the production model
        """
        if technique not in self.models:
            self.models[technique] = {
                'versions': {},
                'production_version': None,
            }

        self.models[technique]['versions'][model_version] = {
            'metrics': metrics,
            'registered_at': datetime.now().isoformat(),
            'is_production': is_production,
        }

        if is_production:
            self.models[technique]['production_version'] = model_version

        self._save_registry()

    def get_production_model(self, technique: str) -> Optional[str]:
        """Get production model version for technique"""
        if technique in self.models:
            return self.models[technique].get('production_version')
        return None

    def compare_models(self, technique: str) -> pd.DataFrame:
        """
        Compare all model versions for a technique

        Args:
            technique: Technique name

        Returns:
            Comparison dataframe
        """
        if technique not in self.models:
            return pd.DataFrame()

        versions = self.models[technique]['versions']

        comparison = []
        for version, data in versions.items():
            metrics = data['metrics']
            comparison.append({
                'version': version,
                'test_score': metrics.get('test_score', 0.0),
                'cv_mean': metrics.get('cv_mean', 0.0),
                'sample_count': metrics.get('sample_count', 0),
                'registered_at': data['registered_at'],
                'is_production': data['is_production'],
            })

        return pd.DataFrame(comparison).sort_values('test_score', ascending=False)

    def promote_to_production(self, technique: str, model_version: str):
        """Promote a model version to production"""
        if technique not in self.models:
            raise ValueError(f"Technique {technique} not found in registry")

        if model_version not in self.models[technique]['versions']:
            raise ValueError(f"Model version {model_version} not found")

        # Demote old production model
        old_prod = self.models[technique]['production_version']
        if old_prod:
            self.models[technique]['versions'][old_prod]['is_production'] = False

        # Promote new model
        self.models[technique]['production_version'] = model_version
        self.models[technique]['versions'][model_version]['is_production'] = True

        self._save_registry()

        print(f"Promoted {technique}/{model_version} to production")
