"""
AI PR Timeline - A plugin to predict PR merge times and CI outcomes using machine learning.

This package provides tools to:
- Collect PR data from GitHub repositories
- Collect CI/test data from GitHub status checks
- Engineer features from PR and CI metadata
- Train machine learning models for PR timelines and CI predictions
- Predict merge times for new PRs
- Predict CI test duration, attempts, and success rates
"""

# Core shared components
from .config import Config
from .data_collection import GitHubDataCollector, TrainingCache

# PR merge time prediction components
from .merge_time import FeatureEngineer, ModelTrainer, PRTimelinePredictor

# CI prediction components  
from .ci import CIFeatureEngineer, CIModelTrainer, CIPredictor

__version__ = "0.2.0"
__author__ = "AI PR Timeline Team"

__all__ = [
    # Core PR prediction components
    "PRTimelinePredictor", "GitHubDataCollector", "ModelTrainer", "FeatureEngineer", 
    "Config", "TrainingCache",
    # CI prediction components
    "CIPredictor", "CIModelTrainer", "CIFeatureEngineer"
]
