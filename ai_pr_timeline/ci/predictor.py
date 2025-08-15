"""
CI prediction module for CI timeline and success estimation.
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..config import Config, DEFAULT_CONFIG
from ..data_collection import GitHubDataCollector
from .model_trainer import CIModelTrainer
from .feature_engineer import CIFeatureEngineer

logger = logging.getLogger(__name__)


class CIPredictor:
    """Main class for predicting CI timelines and success rates."""

    def __init__(self, config: Config = DEFAULT_CONFIG, cache_only: bool = False):
        self.config = config
        self.data_collector = GitHubDataCollector(config, cache_only=cache_only)
        self.duration_trainer = CIModelTrainer(config)
        self.attempts_trainer = CIModelTrainer(config)
        self.success_trainer = CIModelTrainer(config)
        self.feature_engineer = CIFeatureEngineer(config)
        self.trained_models = {}

    def train_models_on_cached_data(self, repo_names: List[str],
                                    hyperparameter_tuning: bool = False) -> Dict:
        """
        Train CI prediction models using only cached data (no API calls).

        Args:
            repo_names: List of repository names to load cached data from
            hyperparameter_tuning: Whether to perform hyperparameter tuning

        Returns:
            Dictionary with training results for all models
        """
        logger.info(f"Training CI models on cached data from {len(repo_names)} repositories")

        all_pr_data = []
        for repo_name in repo_names:
            try:
                cached_prs = self.data_collector.training_cache.get_cached_prs_for_repo(repo_name)
                prs_with_ci = [pr for pr in cached_prs if pr.get('ci_data')]  # Filter to only PRs that have CI data
                if prs_with_ci:
                    all_pr_data.extend(prs_with_ci)
                    logger.info(f"Loaded {len(prs_with_ci)} PRs with CI data from {repo_name}")
                else:
                    logger.warning(f"No PRs with CI data found for {repo_name}")
            except Exception as e:
                logger.warning(f"Could not load cached data for {repo_name}: {e}")

        if not all_pr_data:
            raise ValueError("No cached CI data found. Run collect_data.py --data-type ci first.")

        if len(all_pr_data) < self.config.ci_min_data_points:
            raise ValueError(f"Insufficient CI data: {len(all_pr_data)} PRs found, "
                           f"minimum {self.config.ci_min_data_points} required")

        results = {}

        try:
            logger.info("Training CI duration prediction model")
            X_train, X_test, y_train, y_test = self.duration_trainer.prepare_data(all_pr_data, 'duration')
            self.duration_trainer.train_model(
                X_train, y_train, 
                hyperparameter_tuning=hyperparameter_tuning
            )
            duration_metrics = self.duration_trainer.evaluate_model(X_test, y_test)
            results['duration'] = {
                'metrics': duration_metrics,
                'feature_importance': self.duration_trainer.get_feature_importance().to_dict('records')
            }
            self.trained_models['duration'] = self.duration_trainer
            logger.info(f"Duration model trained. MAE: {duration_metrics['mae']:.4f} hours")
        except Exception as e:
            logger.error(f"Failed to train duration model: {e}")
            results['duration'] = {'error': str(e)}

        try:
            logger.info("Training CI attempts prediction model")
            X_train, X_test, y_train, y_test = self.attempts_trainer.prepare_data(all_pr_data, 'attempts')
            self.attempts_trainer.train_model(
                X_train, y_train,
                hyperparameter_tuning=hyperparameter_tuning
            )
            attempts_metrics = self.attempts_trainer.evaluate_model(X_test, y_test)
            results['attempts'] = {
                'metrics': attempts_metrics,
                'feature_importance': self.attempts_trainer.get_feature_importance().to_dict('records')
            }
            self.trained_models['attempts'] = self.attempts_trainer
            logger.info(f"Attempts model trained. MAE: {attempts_metrics['mae']:.4f} attempts")
        except Exception as e:
            logger.error(f"Failed to train attempts model: {e}")
            results['attempts'] = {'error': str(e)}

        try:
            logger.info("Training CI success rate prediction model")
            X_train, X_test, y_train, y_test = self.success_trainer.prepare_data(all_pr_data, 'success')
            self.success_trainer.train_model(
                X_train, y_train,
                hyperparameter_tuning=hyperparameter_tuning
            )
            success_metrics = self.success_trainer.evaluate_model(X_test, y_test)
            results['success'] = {
                'metrics': success_metrics,
                'feature_importance': self.success_trainer.get_feature_importance().to_dict('records')
            }
            self.trained_models['success'] = self.success_trainer
            logger.info(f"Success model trained. MAE: {success_metrics['mae']:.4f}")
        except Exception as e:
            logger.error(f"Failed to train success model: {e}")
            results['success'] = {'error': str(e)}

        logger.info("CI model training completed")
        return results

    def train_models(self, repo_names: List[str],
                    limit_per_repo: Optional[int] = None,
                    max_new_prs_per_repo: Optional[int] = None,
                    hyperparameter_tuning: bool = False) -> Dict:
        """
        Train CI prediction models on data from repositories.

        Args:
            repo_names: List of repository names
            limit_per_repo: Maximum number of PRs to analyze per repository
            max_new_prs_per_repo: Maximum number of new PRs to fetch from API per repository
            hyperparameter_tuning: Whether to perform hyperparameter tuning

        Returns:
            Dictionary with training results for all models
        """
        logger.info(f"Training CI models on {len(repo_names)} repositories")

        if len(repo_names) == 1:
            _, _ = self.data_collector.collect_all_data(
                repo_names[0], max_new_prs=max_new_prs_per_repo,
                collect_pr_data=True, collect_ci_data=True  # Need PR data for normalization
            )
        else:
            _, _ = self.data_collector.collect_multiple_repos(
                repo_names, max_new_prs_per_repo=max_new_prs_per_repo,
                collect_pr_data=True, collect_ci_data=True  # Need PR data for normalization
            )

        return self.train_models_on_cached_data(repo_names, hyperparameter_tuning)  # Now use the cached data for training

    def predict_ci_timeline(self, repo_name: str, pr_number: int) -> Dict:
        """
        Predict CI timeline and success for a specific PR.

        Args:
            repo_name: Repository name in format 'owner/repo'
            pr_number: PR number

        Returns:
            Dictionary with comprehensive CI predictions
        """
        if not any(self.trained_models.values()):
            raise ValueError("No CI models trained. Train models first.")

        logger.info(f"Predicting CI outcomes for PR #{pr_number} in {repo_name}")

        # Get CI data for this PR
        try:
            ci_data = self._get_pr_ci_data(repo_name, pr_number)
            if not ci_data:
                raise ValueError(f"No CI data found for PR #{pr_number}")

            # Make predictions with all available models
            # Convert list to appropriate format for prediction
            if isinstance(ci_data, list) and len(ci_data) > 0:
                predictions = self._predict_all_models(ci_data[0], repo_name, pr_number)
            else:
                predictions = self._predict_all_models({}, repo_name, pr_number)

            return predictions

        except Exception as e:
            logger.error(f"Error predicting CI outcomes: {e}")
            return {
                'pr_number': pr_number,
                'repository': repo_name,
                'error': str(e),
                'predictions': None
            }

    def predict_batch_ci(self, ci_data_list: List[Dict]) -> List[Dict]:
        """
        Predict CI outcomes for multiple PRs.

        Args:
            ci_data_list: List of CI data dictionaries

        Returns:
            List of prediction results
        """
        if not any(self.trained_models.values()):
            raise ValueError("No CI models trained. Train models first.")

        logger.info(f"Making batch CI predictions for {len(ci_data_list)} items")

        predictions = []
        for ci_data in ci_data_list:
            try:
                pr_predictions = self._predict_all_models(
                    ci_data,
                    ci_data.get('repository', 'unknown'),
                    ci_data.get('pr_number', 0)
                )
                predictions.append(pr_predictions)
            except Exception as e:
                logger.error(f"Error in batch prediction: {e}")
                predictions.append({
                    'pr_number': ci_data.get('pr_number'),
                    'repository': ci_data.get('repository'),
                    'error': str(e),
                    'predictions': None
                })

        return predictions

    def _get_pr_ci_data(self, repo_name: str, pr_number: int) -> List[Dict]:
        """Get CI data for a specific PR."""
        try:
            # For now, we'll need to collect some CI data to make predictions
            # In a real scenario, this could be optimized to get only the specific PR
            _, df = self.data_collector.collect_all_data(repo_name, limit=50, 
                                                        collect_pr_data=False, collect_ci_data=True)
            
            # Filter for the specific PR
            pr_ci_data = df[df['pr_number'] == pr_number]
            
            if pr_ci_data.empty:
                return []

            return [row.to_dict() for _, row in pr_ci_data.iterrows()]

        except Exception as e:
            logger.error(f"Error fetching CI data for PR #{pr_number}: {e}")
            return []

    def _predict_all_models(self, ci_data: Dict, repo_name: str, pr_number: int) -> Dict:
        """Make predictions with all trained models."""
        if isinstance(ci_data, list):
            df = pd.DataFrame(ci_data)
        else:
            df = pd.DataFrame([ci_data])

        predictions = {
            'pr_number': pr_number,
            'repository': repo_name,
            'predictions': {}
        }

        if 'duration' in self.trained_models:
            try:
                duration_pred = self._predict_with_model(df, 'duration')
                predictions['predictions']['duration'] = {
                    'total_duration_hours': float(duration_pred),
                    'total_duration_minutes': float(duration_pred * 60),
                    'confidence': 'medium'  # Could be enhanced with actual confidence intervals
                }
            except Exception as e:
                logger.warning(f"Duration prediction failed: {e}")
                predictions['predictions']['duration'] = {'error': str(e)}

        if 'attempts' in self.trained_models:
            try:
                attempts_pred = self._predict_with_model(df, 'attempts')
                predictions['predictions']['attempts'] = {
                    'expected_attempts': max(1, round(float(attempts_pred))),
                    'confidence': 'medium'
                }
            except Exception as e:
                logger.warning(f"Attempts prediction failed: {e}")
                predictions['predictions']['attempts'] = {'error': str(e)}

        if 'success' in self.trained_models:
            try:
                success_pred = self._predict_with_model(df, 'success')
                predictions['predictions']['success'] = {
                    'success_probability': max(0.0, min(1.0, float(success_pred))),
                    'likely_outcome': 'success' if success_pred > 0.5 else 'failure',
                    'confidence': 'medium'
                }
            except Exception as e:
                logger.warning(f"Success prediction failed: {e}")
                predictions['predictions']['success'] = {'error': str(e)}

        predictions['summary'] = self._generate_prediction_summary(predictions)

        return predictions

    def _predict_with_model(self, df: pd.DataFrame, model_type: str) -> float:
        """Make a prediction with a specific model type."""
        if model_type not in self.trained_models:
            raise ValueError(f"Model {model_type} not trained")

        trainer = self.trained_models[model_type]
        
        X, _ = trainer.feature_engineer.engineer_features(
            df.to_dict('records'), 
            target_type=model_type,
            include_target=False
        )
        
        if X.empty:
            raise ValueError(f"No features generated for {model_type} prediction")

        X_scaled = pd.DataFrame(
            trainer.feature_engineer.scaler.transform(X),
            columns=X.columns
        )

        prediction = trainer.model.predict(X_scaled)[0]
        return float(prediction)

    def _generate_prediction_summary(self, predictions: Dict) -> Dict:
        """Generate a comprehensive summary of all predictions."""
        summary = {}

        if 'duration' in predictions and 'total_duration_hours' in predictions['duration']:
            duration_hours = predictions['duration']['total_duration_hours']
            summary['estimated_completion_time'] = f"{duration_hours:.1f} hours ({duration_hours/24:.1f} days)"

        if 'attempts' in predictions and 'expected_attempts' in predictions['attempts']:
            attempts = predictions['attempts']['expected_attempts']
            summary['expected_retry_attempts'] = attempts
            if attempts > 3:
                summary['retry_risk'] = 'high'
            elif attempts > 1:
                summary['retry_risk'] = 'medium'
            else:
                summary['retry_risk'] = 'low'

        if 'success' in predictions and 'success_probability' in predictions['success']:
            success_prob = predictions['success']['success_probability']
            summary['overall_success_likelihood'] = f"{success_prob*100:.1f}%"
            
            if success_prob > 0.8:
                summary['recommendation'] = 'CI likely to pass with minimal issues'
            elif success_prob > 0.6:
                summary['recommendation'] = 'CI may require some attention'
            elif success_prob > 0.4:
                summary['recommendation'] = 'CI likely to need fixes before passing'
            else:
                summary['recommendation'] = 'CI may require significant fixes'

        return summary

    def get_model_summary(self) -> Dict:
        """Get summary information about trained models."""
        summary = {
            'trained_models': list(self.trained_models.keys()),
            'model_count': len(self.trained_models)
        }

        for model_type, trainer in self.trained_models.items():
            if hasattr(trainer, 'model_metrics'):
                summary[f'{model_type}_metrics'] = trainer.model_metrics

        return summary

    def save_models(self, base_filename: str) -> None:
        """Save all trained CI models."""
        saved_models = []
        
        for model_type, trainer in self.trained_models.items():
            filename = f"{base_filename}_{model_type}.joblib"
            try:
                trainer.save_model(filename)
                saved_models.append(filename)
            except Exception as e:
                logger.error(f"Failed to save {model_type} model: {e}")

        logger.info(f"Saved {len(saved_models)} CI models: {saved_models}")

    def load_models(self, base_filename: str) -> None:
        """Load pre-trained CI models."""
        model_types = ['duration', 'attempts', 'success']
        loaded_models = []

        for model_type in model_types:
            filename = f"{base_filename}_{model_type}.joblib"
            try:
                if model_type == 'duration':
                    self.duration_trainer.load_model(filename)
                    self.trained_models['duration'] = self.duration_trainer
                elif model_type == 'attempts':
                    self.attempts_trainer.load_model(filename)
                    self.trained_models['attempts'] = self.attempts_trainer
                elif model_type == 'success':
                    self.success_trainer.load_model(filename)
                    self.trained_models['success'] = self.success_trainer
                    
                loaded_models.append(model_type)
            except Exception as e:
                logger.warning(f"Could not load {model_type} model from {filename}: {e}")

        logger.info(f"Loaded {len(loaded_models)} CI models: {loaded_models}")

    def get_ci_summary_for_repo(self, repo_name: str) -> Dict:
        """Get CI summary statistics for a repository."""
        try:
            # Load cached PR data with embedded CI data
            cached_prs = self.data_collector.training_cache.get_cached_prs_for_repo(repo_name)
            prs_with_ci = [pr for pr in cached_prs if pr.get('ci_data')]
            
            if not prs_with_ci:
                logger.warning(f"No CI data found for {repo_name}")
                return {}
            
            return self.data_collector.get_ci_summary(prs_with_ci)
        except Exception as e:
            logger.error(f"Error getting CI summary for {repo_name}: {e}")
            return {} 