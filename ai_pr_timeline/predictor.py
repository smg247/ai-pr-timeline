"""
Prediction module for PR timeline estimation.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd

from .config import Config, DEFAULT_CONFIG
from .data_collector import GitHubDataCollector
from .model_trainer import ModelTrainer
from .feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)


class PRTimelinePredictor:
    """Main class for predicting PR merge timelines."""

    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.data_collector = GitHubDataCollector(config)
        self.model_trainer = ModelTrainer(config)
        self.feature_engineer = FeatureEngineer(config)
        self.is_trained = False

    def train_on_repository(self, repo_name: str,
                           model_type: Optional[str] = None, limit: Optional[int] = None,
                           max_new_prs: Optional[int] = None,
                           hyperparameter_tuning: bool = False) -> Dict:
        """
        Train a model on data from a single repository.

        Args:
            repo_name: Repository name in format 'owner/repo'
            model_type: Type of model to train
            limit: Maximum number of PRs to collect
            max_new_prs: Maximum number of new PRs to fetch from API (must be <= limit)
            hyperparameter_tuning: Whether to perform hyperparameter tuning

        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training model on repository: {repo_name}")

        # Collect data
        df = self.data_collector.collect_pr_data(repo_name, limit, max_new_prs)

        if len(df) < self.config.min_data_points:
            raise ValueError(f"Insufficient data: {len(df)} PRs found, "
                           f"minimum {self.config.min_data_points} required")

        # Prepare data
        X_train, X_test, y_train, y_test = self.model_trainer.prepare_data(df)

        # Train model
        self.model_trainer.train_model(
            X_train, y_train,
            model_type=model_type,
            hyperparameter_tuning=hyperparameter_tuning
        )

        # Evaluate model
        metrics = self.model_trainer.evaluate_model(X_test, y_test)
        self.is_trained = True

        # Return comprehensive results
        results = {
            'data_points': len(df),
            'training_size': len(X_train),
            'test_size': len(X_test),
            'metrics': metrics,
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'r2': metrics['r2'],
            'mape': metrics['mape'],
            'feature_importance': self.get_feature_importance(20).to_dict('records')
        }

        logger.info(f"Model training completed. MAE: {metrics['mae']:.2f} hours")
        return results

    def train_on_repositories(self, repo_names: List[str],
                             model_type: Optional[str] = None, limit_per_repo: Optional[int] = None,
                             max_new_prs_per_repo: Optional[int] = None,
                             hyperparameter_tuning: bool = False) -> Dict:
        """
        Train a model on data from multiple repositories.

        Args:
            repo_names: List of repository names
            model_type: Type of model to train
            limit_per_repo: Maximum number of PRs to collect per repository
            max_new_prs_per_repo: Maximum number of new PRs to fetch from API per repository
            hyperparameter_tuning: Whether to perform hyperparameter tuning

        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training model on {len(repo_names)} repositories")

        # Collect data from multiple repositories
        df = self.data_collector.collect_multiple_repos(repo_names, limit_per_repo, max_new_prs_per_repo)

        if len(df) < self.config.min_data_points:
            raise ValueError(f"Insufficient data: {len(df)} PRs found, "
                           f"minimum {self.config.min_data_points} required")

        # Prepare data
        X_train, X_test, y_train, y_test = self.model_trainer.prepare_data(df)

        # Train model
        self.model_trainer.train_model(
            X_train, y_train,
            model_type=model_type,
            hyperparameter_tuning=hyperparameter_tuning
        )

        # Evaluate model
        metrics = self.model_trainer.evaluate_model(X_test, y_test)
        self.is_trained = True

        logger.info(f"Model training completed. MAE: {metrics['mae']:.2f} hours")
        return metrics

    def predict_pr_timeline(self, repo_name: str, pr_number: int) -> Dict:
        """
        Predict timeline for a specific PR.

        Args:
            repo_name: Repository name in format 'owner/repo'
            pr_number: PR number

        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        logger.info(f"Predicting timeline for PR #{pr_number} in {repo_name}")

        # Get PR data
        pr_data = self._get_pr_data(repo_name, pr_number)

        # Make prediction
        prediction = self._predict_single_pr(pr_data)

        return prediction

    def predict_batch(self, pr_data_list: List[Dict]) -> List[Dict]:
        """
        Predict timelines for multiple PRs.

        Args:
            pr_data_list: List of PR data dictionaries

        Returns:
            List of prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        logger.info(f"Making batch predictions for {len(pr_data_list)} PRs")

        predictions = []
        for pr_data in pr_data_list:
            try:
                prediction = self._predict_single_pr(pr_data)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error predicting PR {pr_data.get('pr_number', 'unknown')}: {e}")
                predictions.append({
                    'pr_number': pr_data.get('pr_number'),
                    'error': str(e),
                    'predicted_hours': None,
                    'predicted_days': None
                })

        return predictions

    def _get_pr_data(self, repo_name: str, pr_number: int) -> Dict:
        """Get data for a specific PR."""
        try:
            self.data_collector._log_api_call(f"GET /repos/{repo_name}", f"Getting repository for prediction")
            repo = self.data_collector.github.get_repo(repo_name)
            
            self.data_collector._log_api_call(f"GET /repos/{repo_name}/pulls/{pr_number}", f"Getting PR #{pr_number} for prediction")
            pr = repo.get_pull(pr_number)

            # Extract features similar to data collection
            pr_data = self.data_collector._extract_pr_features(pr)

            return pr_data

        except Exception as e:
            logger.error(f"Error fetching PR data: {e}")
            raise

    def _predict_single_pr(self, pr_data: Dict) -> Dict:
        """Make prediction for a single PR."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame([pr_data])

            # Engineer features (without target)
            features, _ = self.feature_engineer.engineer_features(df, include_target=False)

            # Handle case where we have a trained model but need to ensure feature compatibility
            if hasattr(self.model_trainer, 'feature_names') and self.model_trainer.feature_names:
                # Ensure all required features are present
                for feature_name in self.model_trainer.feature_names:
                    if feature_name not in features.columns:
                        features[feature_name] = 0

                # Select only the features the model was trained on
                features = features[self.model_trainer.feature_names]

            # Scale features (using the same scaler used during training)
            if hasattr(self.feature_engineer, 'scaler') and self.feature_engineer.is_fitted:
                features_scaled = pd.DataFrame(
                    self.feature_engineer.scaler.transform(features),
                    columns=features.columns
                )
            else:
                features_scaled = features

            # Make prediction
            predicted_hours = self.model_trainer.model.predict(features_scaled)[0]  # type: ignore
            predicted_days = predicted_hours / 24  # type: ignore

            result = {
                'pr_number': pr_data.get('pr_number'),
                'repository': pr_data.get('repository'),
                'predicted_hours': float(predicted_hours),  # type: ignore
                'predicted_days': float(predicted_days),  # type: ignore
                'prediction_confidence': self._calculate_confidence(features_scaled)  # type: ignore
            }

            logger.info(f"Predicted {predicted_hours:.1f} hours "
                       f"({predicted_days:.1f} days) for PR #{pr_data.get('pr_number')}")

            return result

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise

    def _calculate_confidence(self, features: pd.DataFrame) -> str:
        """Calculate prediction confidence based on feature values."""
        # Simple confidence calculation based on feature completeness
        non_zero_features = (features != 0).sum().sum()
        total_features = features.size

        if total_features == 0:
            return "low"

        completeness_ratio = non_zero_features / total_features

        if completeness_ratio > 0.7:
            return "high"
        elif completeness_ratio > 0.4:
            return "medium"
        else:
            return "low"

    def get_feature_importance(self, top_k: int = 10) -> pd.DataFrame:
        """Get feature importance from the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")

        return self.model_trainer.get_feature_importance(top_k)

    def save_model(self, filename: str) -> None:
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("No trained model to save")

        self.model_trainer.save_model(filename)
        logger.info(f"Model saved as {filename}")

    def load_model(self, filename: str) -> None:
        """Load a pre-trained model."""
        self.model_trainer.load_model(filename)
        self.feature_engineer = self.model_trainer.feature_engineer
        self.is_trained = True
        logger.info(f"Model loaded from {filename}")

    def get_model_metrics(self) -> Dict:
        """Get metrics from the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting metrics")

        return self.model_trainer.model_metrics
