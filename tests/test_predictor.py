"""
Unit tests for the PRTimelinePredictor class.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from ai_pr_timeline.predictor import PRTimelinePredictor
from ai_pr_timeline.config import Config

class TestPRTimelinePredictor(unittest.TestCase):
    """Test cases for PRTimelinePredictor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.github_token = "test_token"
        self.config.min_data_points = 10
    
    @patch('ai_pr_timeline.predictor.GitHubDataCollector')
    @patch('ai_pr_timeline.predictor.ModelTrainer')
    def test_init(self, mock_trainer, mock_collector):
        """Test predictor initialization."""
        predictor = PRTimelinePredictor(self.config)
        
        self.assertEqual(predictor.config, self.config)
        self.assertFalse(predictor.is_trained)
        mock_collector.assert_called_once_with(self.config)
        mock_trainer.assert_called_once_with(self.config)
    
    def test_categorize_time(self):
        """Test time categorization."""
        predictor = PRTimelinePredictor(self.config)
        
        self.assertEqual(predictor._categorize_time(1), "Very Fast (< 2 hours)")
        self.assertEqual(predictor._categorize_time(12), "Fast (< 1 day)")
        self.assertEqual(predictor._categorize_time(48), "Medium (1-3 days)")
        self.assertEqual(predictor._categorize_time(120), "Slow (3-7 days)")
        self.assertEqual(predictor._categorize_time(200), "Very Slow (> 1 week)")
    
    def test_get_model_info_not_trained(self):
        """Test getting model info when not trained."""
        predictor = PRTimelinePredictor(self.config)
        
        info = predictor.get_model_info()
        
        self.assertEqual(info["status"], "not_trained")
        self.assertIn("message", info)
    
    @patch('ai_pr_timeline.predictor.GitHubDataCollector')
    @patch('ai_pr_timeline.predictor.ModelTrainer')
    def test_get_model_info_trained(self, mock_trainer, mock_collector):
        """Test getting model info when trained."""
        predictor = PRTimelinePredictor(self.config)
        predictor.is_trained = True
        
        # Mock the model trainer
        mock_trainer_instance = mock_trainer.return_value
        mock_trainer_instance.model_metrics = {"mae": 10.0, "r2": 0.8}
        mock_trainer_instance.feature_names = ["feature1", "feature2"]
        predictor.model_trainer = mock_trainer_instance
        
        info = predictor.get_model_info()
        
        self.assertEqual(info["status"], "trained")
        self.assertEqual(info["model_type"], self.config.model_type)
        self.assertEqual(info["feature_count"], 2)
        self.assertEqual(info["metrics"], {"mae": 10.0, "r2": 0.8})
    
    @patch('ai_pr_timeline.predictor.GitHubDataCollector')
    @patch('ai_pr_timeline.predictor.ModelTrainer')
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
            predictor.predict_pr_timeline(pr_data)
        
        self.assertIn("Model must be trained", str(context.exception))
    
    def test_train_on_data_insufficient_data(self):
        """Test training with insufficient data."""
        predictor = PRTimelinePredictor(self.config)
        
        # Create small dataset
        df = pd.DataFrame({
            'merge_time_hours': [10, 20],
            'title': ['PR 1', 'PR 2'],
            'body': ['Description 1', 'Description 2'],
            'review_count': [1, 2],
            'comment_count': [0, 1],
            'commit_count': [1, 2],
            'files_changed': [2, 3],
            'additions': [50, 100],
            'deletions': [10, 20],
            'created_hour': [10, 14],
            'created_day': [1, 3],
            'author_association': ['MEMBER', 'CONTRIBUTOR'],
            'is_draft': [False, False]
        })
        
        with self.assertRaises(ValueError) as context:
            predictor.train_on_data(df)
        
        self.assertIn("Insufficient data", str(context.exception))

if __name__ == '__main__':
    unittest.main() 