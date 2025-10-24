"""
Feedback processing and ML training pipeline
Modular architecture for training separate models per technique
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from pathlib import Path
from app.config import settings


class FeedbackProcessor:
    """
    Process user feedback and prepare training datasets
    Maintains separate datasets for each prediction technique
    """

    def __init__(self, db_session: Session):
        self.db = db_session
        self.model_path = Path(settings.ml_model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)

    def collect_progression_feedback(self) -> pd.DataFrame:
        """
        Collect and process progression feedback data

        Returns:
            DataFrame ready for training
        """
        from app.models.predictions import ProgressionFeedback, ProgressionPrediction

        # Query feedback
        feedbacks = self.db.query(ProgressionFeedback).all()

        records = []
        for fb in feedbacks:
            prog_pred = fb.progression_prediction
            prediction = prog_pred.prediction

            # Extract features
            feature_dict = {
                # Progression features
                'progressed_planet': prog_pred.progressed_planet,
                'natal_planet': prog_pred.natal_planet,
                'aspect_type': prog_pred.aspect_type,
                'orb': prog_pred.orb,

                # Prediction metadata
                'predicted_category': prediction.predicted_event_category,
                'confidence_score': prediction.confidence_score,

                # Feedback (target variables)
                'event_occurred': fb.event_occurred,
                'accuracy_rating': fb.accuracy_rating,
                'relevance_rating': fb.relevance_rating,
                'date_accuracy_days': fb.date_accuracy_days,
            }

            records.append(feature_dict)

        return pd.DataFrame(records)

    def collect_transit_feedback(self) -> pd.DataFrame:
        """Collect transit feedback data"""
        from app.models.predictions import TransitFeedback, TransitPrediction

        feedbacks = self.db.query(TransitFeedback).all()

        records = []
        for fb in feedbacks:
            transit_pred = fb.transit_prediction
            prediction = transit_pred.prediction

            feature_dict = {
                'transiting_planet': transit_pred.transiting_planet,
                'natal_planet': transit_pred.natal_planet,
                'aspect_type': transit_pred.aspect_type,
                'orb': transit_pred.orb,
                'is_retrograde': transit_pred.is_retrograde,

                'predicted_category': prediction.predicted_event_category,
                'confidence_score': prediction.confidence_score,

                'event_occurred': fb.event_occurred,
                'accuracy_rating': fb.accuracy_rating,
                'intensity_rating': fb.intensity_rating,
                'date_accuracy_days': fb.date_accuracy_days,
            }

            records.append(feature_dict)

        return pd.DataFrame(records)

    def collect_zr_feedback(self) -> pd.DataFrame:
        """Collect Zodiacal Releasing feedback data"""
        from app.models.predictions import ZRFeedback, ZodiacalReleasingPrediction

        feedbacks = self.db.query(ZRFeedback).all()

        records = []
        for fb in feedbacks:
            zr_pred = fb.zr_prediction
            prediction = zr_pred.prediction

            feature_dict = {
                'period_lord': zr_pred.period_lord,
                'sub_period_lord': zr_pred.sub_period_lord,
                'level': zr_pred.level,
                'is_peak_period': zr_pred.is_peak_period,
                'is_loosening_period': zr_pred.is_loosening_period,

                'predicted_category': prediction.predicted_event_category,
                'confidence_score': prediction.confidence_score,

                'event_occurred': fb.event_occurred,
                'accuracy_rating': fb.accuracy_rating,
                'theme_accuracy_rating': fb.theme_accuracy_rating,
                'date_accuracy_days': fb.date_accuracy_days,
            }

            records.append(feature_dict)

        return pd.DataFrame(records)

    def collect_lb_feedback(self) -> pd.DataFrame:
        """Collect Loosening of Bonds feedback data"""
        from app.models.predictions import LBFeedback, LooseningBondsPrediction

        feedbacks = self.db.query(LBFeedback).all()

        records = []
        for fb in feedbacks:
            lb_pred = fb.lb_prediction
            prediction = lb_pred.prediction

            feature_dict = {
                'bond_planet': lb_pred.bond_planet,
                'bond_type': lb_pred.bond_type,
                'intensity_score': lb_pred.intensity_score,

                'predicted_category': prediction.predicted_event_category,
                'confidence_score': prediction.confidence_score,

                'event_occurred': fb.event_occurred,
                'accuracy_rating': fb.accuracy_rating,
                'refinement_value': fb.refinement_value,
                'date_accuracy_days': fb.date_accuracy_days,
            }

            records.append(feature_dict)

        return pd.DataFrame(records)

    def prepare_features(self, df: pd.DataFrame, technique: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for ML training

        Args:
            df: Raw feedback dataframe
            technique: Technique name (for technique-specific encoding)

        Returns:
            Tuple of (X_features, y_target)
        """
        # Make copy
        data = df.copy()

        # One-hot encode categorical variables
        categorical_cols = data.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            if col not in ['event_occurred']:  # Don't encode target
                dummies = pd.get_dummies(data[col], prefix=col)
                data = pd.concat([data, dummies], axis=1)
                data.drop(col, axis=1, inplace=True)

        # Separate features and target
        target_col = 'event_occurred'
        if target_col in data.columns:
            y = data[target_col].astype(int).values
            X = data.drop(target_col, axis=1).values
        else:
            # No target available
            y = np.array([])
            X = data.values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        return X, y

    def calculate_technique_performance(self, technique: str) -> Dict:
        """
        Calculate performance metrics for a specific technique

        Args:
            technique: Technique name

        Returns:
            Performance metrics dictionary
        """
        # Collect feedback
        if technique == 'progressions':
            df = self.collect_progression_feedback()
        elif technique == 'transits':
            df = self.collect_transit_feedback()
        elif technique == 'zr':
            df = self.collect_zr_feedback()
        elif technique == 'lb':
            df = self.collect_lb_feedback()
        else:
            raise ValueError(f"Unknown technique: {technique}")

        if len(df) == 0:
            return {
                'technique': technique,
                'sample_count': 0,
                'accuracy': 0.0,
                'avg_date_error_days': 0.0,
                'status': 'insufficient_data'
            }

        # Calculate metrics
        total = len(df)
        correct = df['event_occurred'].sum()
        accuracy = correct / total if total > 0 else 0.0

        avg_accuracy_rating = df['accuracy_rating'].mean()

        # Date accuracy
        date_errors = df['date_accuracy_days'].dropna()
        avg_date_error = date_errors.mean() if len(date_errors) > 0 else 0.0

        return {
            'technique': technique,
            'sample_count': total,
            'accuracy': accuracy,
            'avg_accuracy_rating': avg_accuracy_rating,
            'avg_date_error_days': avg_date_error,
            'status': 'active' if total >= 10 else 'insufficient_data'
        }

    def compare_technique_performance(self) -> pd.DataFrame:
        """
        Compare performance across all techniques
        This helps identify which techniques should be refined or removed

        Returns:
            DataFrame with comparative metrics
        """
        techniques = ['progressions', 'transits', 'zr', 'lb']

        results = []
        for technique in techniques:
            metrics = self.calculate_technique_performance(technique)
            results.append(metrics)

        comparison_df = pd.DataFrame(results)

        # Add recommendations
        comparison_df['recommendation'] = comparison_df.apply(
            self._get_recommendation, axis=1
        )

        return comparison_df

    def _get_recommendation(self, row: pd.Series) -> str:
        """Generate recommendation for technique based on performance"""
        if row['sample_count'] < 10:
            return 'COLLECT_MORE_DATA'
        elif row['accuracy'] >= 0.7:
            return 'EXCELLENT_KEEP'
        elif row['accuracy'] >= 0.5:
            return 'GOOD_REFINE'
        elif row['accuracy'] >= 0.3:
            return 'POOR_MAJOR_REFINEMENT'
        else:
            return 'REMOVE_OR_REDESIGN'

    def export_training_data(self, technique: str, output_path: Optional[Path] = None):
        """
        Export training data for a technique

        Args:
            technique: Technique name
            output_path: Optional custom output path
        """
        # Collect data
        if technique == 'progressions':
            df = self.collect_progression_feedback()
        elif technique == 'transits':
            df = self.collect_transit_feedback()
        elif technique == 'zr':
            df = self.collect_zr_feedback()
        elif technique == 'lb':
            df = self.collect_lb_feedback()
        else:
            raise ValueError(f"Unknown technique: {technique}")

        # Default path
        if output_path is None:
            data_path = Path(settings.training_data_path)
            data_path.mkdir(parents=True, exist_ok=True)
            output_path = data_path / f"{technique}_training_data_{datetime.now().strftime('%Y%m%d')}.csv"

        # Export
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} records to {output_path}")

        return output_path
