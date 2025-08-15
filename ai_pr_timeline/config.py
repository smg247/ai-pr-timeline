"""
Configuration settings for the AI PR Timeline plugin.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration settings for the PR timeline predictor."""

    # GitHub API settings
    github_token: Optional[str] = None
    github_base_url: str = "https://api.github.com"

    # Model settings
    model_type: str = "random_forest"  # Options: random_forest, xgboost, lightgbm
    test_size: float = 0.2
    random_state: int = 42

    # Feature engineering settings
    include_text_features: bool = True
    max_text_features: int = 50  # Reduced from 1000 to balance with structural features
    text_feature_weight: float = 0.3  # Weight factor for text features (0.0-1.0)

    # Data collection settings
    max_prs_per_repo: int = 1000
    min_data_points: int = 50

    # CI-specific settings
    ci_model_type: str = "random_forest"  # Options: random_forest, xgboost, lightgbm
    ci_test_size: float = 0.2
    ci_min_data_points: int = 50
    max_ci_runs_per_pr: int = 100  # Maximum CI runs to analyze per PR
    ci_success_threshold: float = 0.95  # Success rate threshold for CI reliability
    include_ci_text_features: bool = True
    max_ci_text_features: int = 30
    ci_text_feature_weight: float = 0.2

    # File paths
    data_dir: str = "data"
    model_dir: str = "models"
    cache_dir: str = "cache"
    training_cache_dir: str = "training_cache"

    def __post_init__(self):
        """Load configuration from environment variables."""
        self.github_token = os.getenv("GITHUB_TOKEN", self.github_token)
    
    def ensure_directories(self):
        """Create directories if they don't exist. Call this before using the directories."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.training_cache_dir, exist_ok=True)


# Default configuration instance
DEFAULT_CONFIG = Config()
