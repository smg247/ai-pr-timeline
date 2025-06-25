"""
AI PR Timeline - A plugin to predict PR merge times using machine learning.

This package provides tools to:
- Collect PR data from GitHub repositories
- Engineer features from PR metadata
- Train machine learning models
- Predict merge times for new PRs
"""

from .config import Config
from .data_collector import GitHubDataCollector
from .feature_engineer import FeatureEngineer
from .model_trainer import ModelTrainer
from .predictor import PRTimelinePredictor
from .training_cache import TrainingCache

__version__ = "0.1.0"
__author__ = "AI PR Timeline Team"

__all__ = ["PRTimelinePredictor", "GitHubDataCollector", "ModelTrainer", "FeatureEngineer", "Config", "TrainingCache"]
