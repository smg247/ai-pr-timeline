"""
CI model training module for CI prediction.
"""

import logging
import joblib
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb

from ..config import Config, DEFAULT_CONFIG
from .feature_engineer import CIFeatureEngineer

logger = logging.getLogger(__name__)


class CIModelTrainer:
    """Trains and evaluates ML models for CI prediction."""

    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.model = None
        self.feature_engineer = CIFeatureEngineer(config)
        self.feature_names = None
        self.model_metrics = {}
        self.target_type = 'duration'  # duration, attempts, or success

    def prepare_data(self, df: pd.DataFrame, target_type: str = 'duration') -> Tuple[pd.DataFrame, pd.DataFrame,
                                                                                   pd.Series, pd.Series]:
        """
        Prepare data for CI model training.

        Args:
            df: Raw CI data DataFrame
            target_type: Type of target to predict ('duration', 'attempts', 'success')

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Preparing CI data for training, target: {target_type}")
        self.target_type = target_type

        # Engineer features
        X, y = self.feature_engineer.engineer_features(df, target_type=target_type)
        
        if X.empty or y is None or len(y) == 0:
            raise ValueError("No valid data after feature engineering")

        self.feature_names = list(X.columns)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.ci_test_size,
            random_state=self.config.random_state,
            stratify=None
        )
        X_train = pd.DataFrame(X_train)  # type: ignore
        X_test = pd.DataFrame(X_test)  # type: ignore
        y_train = pd.Series(y_train)  # type: ignore
        y_test = pd.Series(y_test)  # type: ignore

        # Scale features
        X_train_scaled, X_test_scaled = self.feature_engineer.scale_features(X_train, X_test)

        logger.info(f"CI training set size: {len(X_train_scaled)}")
        logger.info(f"CI test set size: {len(X_test_scaled)}")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   model_type: Optional[str] = None, hyperparameter_tuning: bool = False) -> None:
        """
        Train a CI prediction model.

        Args:
            X_train: Training features
            y_train: Training targets
            model_type: Type of model to train ('random_forest', 'xgboost', 'lightgbm')
            hyperparameter_tuning: Whether to perform hyperparameter tuning
        """
        model_type = model_type or self.config.ci_model_type
        logger.info(f"Training CI {model_type} model for {self.target_type} prediction")

        if model_type == 'random_forest':
            self.model = self._train_random_forest(X_train, y_train, hyperparameter_tuning)
        elif model_type == 'xgboost':
            self.model = self._train_xgboost(X_train, y_train, hyperparameter_tuning)
        elif model_type == 'lightgbm':
            self.model = self._train_lightgbm(X_train, y_train, hyperparameter_tuning)
        else:
            raise ValueError(f"Unsupported CI model type: {model_type}")

        logger.info("CI model training completed")

    def _train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                           tune_hyperparams: bool) -> RandomForestRegressor:
        """Train a Random Forest model for CI prediction."""
        if tune_hyperparams:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

            rf = RandomForestRegressor(random_state=self.config.random_state)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5,
                scoring='neg_mean_absolute_error',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_
        else:
            # Adjusted parameters for CI data
            rf = RandomForestRegressor(
                n_estimators=150,  # Slightly less than PR model
                max_depth=15,      # Slightly less depth
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            return rf

    def _train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                      tune_hyperparams: bool) -> xgb.XGBRegressor:
        """Train an XGBoost model for CI prediction."""
        if tune_hyperparams:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }

            xgb_model = xgb.XGBRegressor(random_state=self.config.random_state)
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=5,
                scoring='neg_mean_absolute_error',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_
        else:
            xgb_model = xgb.XGBRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.config.random_state
            )
            xgb_model.fit(X_train, y_train)
            return xgb_model

    def _train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                       tune_hyperparams: bool) -> lgb.LGBMRegressor:
        """Train a LightGBM model for CI prediction."""
        if tune_hyperparams:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 62, 124]
            }

            lgb_model = lgb.LGBMRegressor(random_state=self.config.random_state)
            grid_search = GridSearchCV(
                lgb_model, param_grid, cv=5,
                scoring='neg_mean_absolute_error',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_
        else:
            lgb_model = lgb.LGBMRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                num_leaves=31,
                random_state=self.config.random_state
            )
            lgb_model.fit(X_train, y_train)
            return lgb_model

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the trained CI model.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")

        logger.info("Evaluating CI model")

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate regression metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Calculate MAPE (handle division by zero)
        non_zero_mask = y_test != 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100  # type: ignore
        else:
            mape = np.inf

        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'target_type': self.target_type
        }

        # Target-specific metrics
        if self.target_type == 'duration':
            # Convert back to minutes for interpretability
            metrics['mae_minutes'] = mae * 60
            metrics['rmse_minutes'] = rmse * 60
        elif self.target_type == 'success':
            # For success rate prediction, add classification-like metrics
            # Convert predictions to binary (>0.5 = success, <0.5 = failure)
            y_pred_binary = (y_pred > 0.5).astype(int)  # type: ignore
            y_test_binary = (y_test > 0.5).astype(int)
            metrics['accuracy'] = accuracy_score(y_test_binary, y_pred_binary)

        self.model_metrics = metrics

        logger.info("CI model evaluation metrics:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric.upper()}: {value:.4f}")

        return metrics

    def get_feature_importance(self, top_k: int = 15) -> pd.DataFrame:
        """
        Get feature importance from the trained CI model.

        Args:
            top_k: Number of top features to return

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importance")

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            raise ValueError("Model does not support feature importance")

        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return feature_importance_df.head(top_k)

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation on the CI model.

        Args:
            X: Features
            y: Targets
            cv: Number of cross-validation folds

        Returns:
            Dictionary with cross-validation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before cross-validation")

        logger.info(f"Performing {cv}-fold cross-validation on CI model")

        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model, X, y,
            cv=cv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )

        cv_metrics = {
            'cv_mae_mean': -cv_scores.mean(),
            'cv_mae_std': cv_scores.std(),
            'cv_scores': cv_scores
        }

        logger.info(f"Cross-validation MAE: {cv_metrics['cv_mae_mean']:.4f} "
                   f"± {cv_metrics['cv_mae_std']:.4f}")

        return cv_metrics

    def save_model(self, filename: str) -> None:
        """Save the trained CI model and feature engineer."""
        if self.model is None:
            raise ValueError("No CI model to save")

        model_path = f"{self.config.model_dir}/{filename}"

        # Create a sanitized config without sensitive information
        sanitized_config = Config(
            github_token=None,  # Remove the token for security
            github_base_url=self.config.github_base_url,
            model_type=self.config.model_type,
            ci_model_type=self.config.ci_model_type,
            test_size=self.config.test_size,
            ci_test_size=self.config.ci_test_size,
            random_state=self.config.random_state,
            include_text_features=self.config.include_text_features,
            include_ci_text_features=self.config.include_ci_text_features,
            max_text_features=self.config.max_text_features,
            max_ci_text_features=self.config.max_ci_text_features,
            max_prs_per_repo=self.config.max_prs_per_repo,
            min_data_points=self.config.min_data_points,
            ci_min_data_points=self.config.ci_min_data_points,
            data_dir=self.config.data_dir,
            model_dir=self.config.model_dir,
            cache_dir=self.config.cache_dir
        )

        # Save model and associated objects
        model_data = {
            'model': self.model,
            'feature_engineer': self.feature_engineer,
            'feature_names': self.feature_names,
            'config': sanitized_config,
            'metrics': self.model_metrics,
            'target_type': self.target_type
        }

        joblib.dump(model_data, model_path)
        logger.info(f"CI model saved to {model_path}")

    def load_model(self, filename: str) -> None:
        """Load a trained CI model and feature engineer."""
        model_path = f"{self.config.model_dir}/{filename}"

        model_data = joblib.load(model_path)

        self.model = model_data['model']
        self.feature_engineer = model_data['feature_engineer']
        self.feature_names = model_data['feature_names']
        self.model_metrics = model_data.get('metrics', {})
        self.target_type = model_data.get('target_type', 'duration')

        logger.info(f"CI model loaded from {model_path}")
        logger.info(f"Target type: {self.target_type}") 