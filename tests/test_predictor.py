"""
Unit tests for the PRTimelinePredictor class.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from ai_pr_timeline import PRTimelinePredictor
from ai_pr_timeline.config import Config

class TestPRTimelinePredictor(unittest.TestCase):
    """Test cases for PRTimelinePredictor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.github_token = "test_token"
        self.config.min_data_points = 10
    
    @patch('ai_pr_timeline.merge_time.predictor.GitHubDataCollector')
    @patch('ai_pr_timeline.merge_time.predictor.ModelTrainer')
    def test_init(self, mock_trainer, mock_collector):
        """Test predictor initialization."""
        predictor = PRTimelinePredictor(self.config)
        
        self.assertEqual(predictor.config, self.config)
        self.assertFalse(predictor.is_trained)
        mock_collector.assert_called_once_with(self.config, cache_only=False)
        mock_trainer.assert_called_once_with(self.config)
    
    def test_get_model_metrics(self):
        """Test getting model metrics."""
        predictor = PRTimelinePredictor(self.config)
        
        with self.assertRaises(ValueError):
            predictor.get_model_metrics()
    
    @patch('ai_pr_timeline.merge_time.predictor.GitHubDataCollector')
    @patch('ai_pr_timeline.merge_time.predictor.ModelTrainer')
    def test_predict_pr_timeline_not_trained(self, mock_trainer, mock_collector):
        """Test prediction when model is not trained."""
        predictor = PRTimelinePredictor(self.config)
        
        pr_data = {
            "title": "Test PR",
            "body": "Test description",
            "review_count": 2,
            "comment_count": 5,
            "commit_count": 3,
            "files_changed": 4,
            "additions": 100,
            "deletions": 20,
            "created_hour": 14,
            "created_day": 2,
            "author_association": "MEMBER",
            "is_draft": False
        }
        
        with self.assertRaises(ValueError) as context:
            predictor.predict_pr_timeline("test/repo", 123)
        
        self.assertIn("Model must be trained", str(context.exception))

if __name__ == '__main__':
    unittest.main() 