"""
Main predictor interface for PR timeline estimation.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime

from .config import Config, DEFAULT_CONFIG
from .data_collector import GitHubDataCollector
from .model_trainer import ModelTrainer

logger = logging.getLogger(__name__)

class PRTimelinePredictor:
    """Main interface for predicting PR merge times."""
    
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.data_collector = GitHubDataCollector(config)
        self.model_trainer = ModelTrainer(config)
        self.is_trained = False
    
    def train_on_repository(self, repo_name: str, save_model: bool = True, 
                          model_filename: str = None) -> Dict[str, Any]:
        """
        Train the model on data from a specific repository.
        
        Args:
            repo_name: Repository name in format 'owner/repo'
            save_model: Whether to save the trained model
            model_filename: Filename for saving the model
            
        Returns:
            Dictionary with training results and metrics
        """
        logger.info(f"Training model on repository: {repo_name}")
        
        # Collect data
        df = self.data_collector.collect_pr_data(repo_name)
        
        if len(df) < self.config.min_data_points:
            raise ValueError(f"Insufficient data: {len(df)} PRs (minimum: {self.config.min_data_points})")
        
        # Train model
        results = self.train_on_data(df, save_model, model_filename)
        results['repository'] = repo_name
        results['data_points'] = len(df)
        
        return results
    
    def train_on_repositories(self, repo_names: List[str], save_model: bool = True,
                            model_filename: str = None) -> Dict[str, Any]:
        """
        Train the model on data from multiple repositories.
        
        Args:
            repo_names: List of repository names in format 'owner/repo'
            save_model: Whether to save the trained model
            model_filename: Filename for saving the model
            
        Returns:
            Dictionary with training results and metrics
        """
        logger.info(f"Training model on {len(repo_names)} repositories")
        
        # Collect data from all repositories
        df = self.data_collector.collect_multiple_repos(repo_names)
        
        if len(df) < self.config.min_data_points:
            raise ValueError(f"Insufficient data: {len(df)} PRs (minimum: {self.config.min_data_points})")
        
        # Train model
        results = self.train_on_data(df, save_model, model_filename)
        results['repositories'] = repo_names
        results['data_points'] = len(df)
        
        return results
    
    def train_on_data(self, df: pd.DataFrame, save_model: bool = True,
                     model_filename: str = None) -> Dict[str, Any]:
        """
        Train the model on provided data.
        
        Args:
            df: DataFrame with PR data
            save_model: Whether to save the trained model
            model_filename: Filename for saving the model
            
        Returns:
            Dictionary with training results and metrics
        """
        logger.info("Training model on provided data")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.model_trainer.prepare_data(df)
        
        # Train model
        self.model_trainer.train_model(X_train, y_train)
        
        # Evaluate model
        metrics = self.model_trainer.evaluate_model(X_test, y_test)
        
        # Get feature importance
        feature_importance = self.model_trainer.get_feature_importance()
        
        # Cross-validation
        cv_metrics = self.model_trainer.cross_validate(
            pd.concat([X_train, X_test]), 
            pd.concat([y_train, y_test])
        )
        
        # Save model if requested
        if save_model:
            filename = model_filename or f"pr_timeline_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            self.model_trainer.save_model(filename)
        
        self.is_trained = True
        
        results = {
            'metrics': metrics,
            'cv_metrics': cv_metrics,
            'feature_importance': feature_importance.to_dict('records'),
            'training_size': len(X_train),
            'test_size': len(X_test)
        }
        
        return results
    
    def load_trained_model(self, model_filename: str) -> None:
        """
        Load a pre-trained model.
        
        Args:
            model_filename: Filename of the saved model
        """
        self.model_trainer.load_model(model_filename)
        self.is_trained = True
        logger.info(f"Loaded trained model: {model_filename}")
    
    def predict_pr_timeline(self, pr_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict merge timeline for a single PR.
        
        Args:
            pr_data: Dictionary with PR features
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained or loaded before making predictions")
        
        # Convert to DataFrame
        df = pd.DataFrame([pr_data])
        
        # Engineer features
        X, _ = self.model_trainer.feature_engineer.engineer_features(df)
        
        # Scale features
        X_scaled, _ = self.model_trainer.feature_engineer.scale_features(X)
        
        # Make prediction
        prediction_hours = self.model_trainer.model.predict(X_scaled)[0]
        
        # Convert to more interpretable formats
        prediction_days = prediction_hours / 24
        
        # Calculate confidence intervals (rough estimate)
        # This is a simplified approach - in production, you might want to use more sophisticated methods
        std_error = np.std(self.model_trainer.model_metrics.get('cv_scores', [0])) if hasattr(self, 'cv_scores') else prediction_hours * 0.2
        
        result = {
            'predicted_hours': round(prediction_hours, 2),
            'predicted_days': round(prediction_days, 2),
            'confidence_interval_hours': {
                'lower': round(max(0, prediction_hours - 1.96 * std_error), 2),
                'upper': round(prediction_hours + 1.96 * std_error, 2)
            },
            'time_category': self._categorize_time(prediction_hours)
        }
        
        return result
    
    def predict_from_github_pr(self, repo_name: str, pr_number: int) -> Dict[str, Any]:
        """
        Predict merge timeline for a PR directly from GitHub.
        
        Args:
            repo_name: Repository name in format 'owner/repo'
            pr_number: PR number
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained or loaded before making predictions")
        
        # Get PR data from GitHub
        repo = self.data_collector.github.get_repo(repo_name)
        pr = repo.get_pull(pr_number)
        
        # Extract features
        pr_data = self.data_collector._extract_pr_features(pr)
        
        # Remove target variable if present
        if 'merge_time_hours' in pr_data:
            del pr_data['merge_time_hours']
        
        # Make prediction
        result = self.predict_pr_timeline(pr_data)
        result['pr_number'] = pr_number
        result['repository'] = repo_name
        result['pr_title'] = pr.title
        
        return result
    
    def _categorize_time(self, hours: float) -> str:
        """Categorize prediction into time buckets."""
        if hours < 2:
            return "Very Fast (< 2 hours)"
        elif hours < 24:
            return "Fast (< 1 day)"
        elif hours < 72:
            return "Medium (1-3 days)"
        elif hours < 168:  # 1 week
            return "Slow (3-7 days)"
        else:
            return "Very Slow (> 1 week)"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if not self.is_trained:
            return {"status": "not_trained", "message": "No model has been trained or loaded"}
        
        info = {
            "status": "trained",
            "model_type": self.config.model_type,
            "metrics": self.model_trainer.model_metrics,
            "feature_count": len(self.model_trainer.feature_names) if self.model_trainer.feature_names else 0
        }
        
        return info
    
    def benchmark_predictions(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Benchmark the model against test data.
        
        Args:
            test_data: DataFrame with test PR data including actual merge times
            
        Returns:
            Dictionary with benchmark results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained or loaded before benchmarking")
        
        predictions = []
        actuals = []
        
        for _, row in test_data.iterrows():
            actual_time = row['merge_time_hours']
            pr_data = row.drop('merge_time_hours').to_dict()
            
            try:
                pred_result = self.predict_pr_timeline(pr_data)
                predictions.append(pred_result['predicted_hours'])
                actuals.append(actual_time)
            except Exception as e:
                logger.warning(f"Error predicting for PR: {e}")
                continue
        
        if not predictions:
            return {"error": "No successful predictions made"}
        
        # Calculate benchmark metrics
        mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
        mse = np.mean((np.array(predictions) - np.array(actuals)) ** 2)
        rmse = np.sqrt(mse)
        
        benchmark_results = {
            "sample_size": len(predictions),
            "mae": round(mae, 2),
            "mse": round(mse, 2),
            "rmse": round(rmse, 2),
            "predictions": predictions[:10],  # First 10 predictions
            "actuals": actuals[:10]  # First 10 actual values
        }
        
        return benchmark_results 