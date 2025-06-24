"""
AI PR Timeline Prediction Plugin

A Python plugin for predicting pull request merge times using machine learning
and historical GitHub data.
"""

__version__ = "0.1.0"
__author__ = "AI PR Timeline Team"

from .predictor import PRTimelinePredictor
from .data_collector import GitHubDataCollector
from .model_trainer import ModelTrainer

__all__ = ["PRTimelinePredictor", "GitHubDataCollector", "ModelTrainer"] 