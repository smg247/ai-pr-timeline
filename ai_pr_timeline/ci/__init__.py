"""
CI Prediction Module

This module contains all components for predicting CI outcomes:
- Feature engineering for CI data
- Model training for CI predictions (duration, attempts, success)
- Prediction interface for CI analysis
"""

from .feature_engineer import CIFeatureEngineer
from .model_trainer import CIModelTrainer
from .predictor import CIPredictor

__all__ = [
    "CIFeatureEngineer",
    "CIModelTrainer",
    "CIPredictor"
] 