"""
CI feature engineering module for CI prediction.
"""

import logging
from typing import Tuple, Optional, Dict, List, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from scipy.sparse import csr_matrix

from ..config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class CIFeatureEngineer:
    """Handles feature engineering for CI prediction data."""

    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.tfidf_vectorizer = None
        self.is_fitted = False

    def engineer_features(self, df: pd.DataFrame,
                         target_type: str = 'duration',
                         include_target: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Engineer features from raw CI data.

        Args:
            df: Raw CI data DataFrame
            target_type: Type of target to predict ('duration', 'attempts', 'success')
            include_target: Whether to include target variable

        Returns:
            Tuple of (features_df, target_series) or (features_df, None)
        """
        logger.info(f"Engineering CI features for target: {target_type}")

        if df.empty:
            return pd.DataFrame(), None

        # Make a copy to avoid modifying original data
        data = df.copy()

        # Aggregate CI data by PR for prediction
        pr_aggregated = self._aggregate_ci_by_pr(data)

        # Extract basic features
        features = self._extract_basic_features(pr_aggregated)

        # Extract CI-specific features
        ci_features = self._extract_ci_features(pr_aggregated)
        features = pd.concat([features, ci_features], axis=1)

        # Text features (optional)
        if self.config.include_ci_text_features:
            text_features = self._extract_text_features(pr_aggregated)
            if text_features is not None and not text_features.empty:
                features = pd.concat([features, text_features], axis=1)

        # Handle categorical variables
        features = self._encode_categorical_features(features)

        # Handle missing values
        features = self._handle_missing_values(features)

        # Balance feature importance
        features = self._balance_feature_weights(features)

        logger.info(f"Engineered {len(features.columns)} CI features")

        # Extract target variable if requested
        target = None
        if include_target:
            target = self._extract_target(pr_aggregated, target_type)

        return features, target

    def _aggregate_ci_by_pr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate CI data by PR number for prediction.

        Args:
            df: Raw CI data DataFrame

        Returns:
            DataFrame with aggregated CI data per PR
        """
        if df.empty:
            return pd.DataFrame()

        # Group by PR and repository
        groupby_cols = ['repository', 'pr_number']
        pr_groups = df.groupby(groupby_cols)

        aggregated_data = []

        for key, group in pr_groups:
            if isinstance(key, tuple) and len(key) == 2:
                repo, pr_num = key
            else:
                continue
            pr_data = self._aggregate_single_pr(group)
            pr_data['repository'] = repo
            pr_data['pr_number'] = pr_num
            aggregated_data.append(pr_data)

        return pd.DataFrame(aggregated_data)

    def _aggregate_single_pr(self, pr_ci_data: pd.DataFrame) -> Dict:
        """Aggregate CI data for a single PR."""
        if pr_ci_data.empty:
            return {}

        # Basic PR info (should be the same for all CI runs)
        first_row = pr_ci_data.iloc[0]
        
        aggregated = {
            'pr_title': first_row.get('pr_title', ''),
            'pr_created_at': first_row.get('pr_created_at'),
            'pr_merged_at': first_row.get('pr_merged_at'),
            'pr_state': first_row.get('pr_state'),
            'pr_files_changed': first_row.get('pr_files_changed', 0),
            'pr_additions': first_row.get('pr_additions', 0),
            'pr_deletions': first_row.get('pr_deletions', 0),
            'pr_commits': first_row.get('pr_commits', 0),
            'pr_author': first_row.get('pr_author'),
            'pr_is_draft': first_row.get('pr_is_draft', False),
        }

        # CI-specific aggregations
        ci_runs = pr_ci_data[pr_ci_data['ci_duration_seconds'].notna()]
        
        if not ci_runs.empty:
            # Duration statistics
            aggregated['ci_total_duration'] = ci_runs['ci_duration_seconds'].sum()
            aggregated['ci_avg_duration'] = ci_runs['ci_duration_seconds'].mean()
            aggregated['ci_max_duration'] = ci_runs['ci_duration_seconds'].max()
            aggregated['ci_min_duration'] = ci_runs['ci_duration_seconds'].min()

            # Test counts and success rates
            aggregated['ci_total_runs'] = len(pr_ci_data)
            aggregated['ci_unique_tests'] = pr_ci_data['ci_name'].nunique()
            aggregated['ci_successful_runs'] = len(pr_ci_data[pr_ci_data['ci_state'] == 'success'])
            aggregated['ci_failed_runs'] = len(pr_ci_data[pr_ci_data['ci_state'].isin(['failure', 'error'])])
            aggregated['ci_success_rate'] = aggregated['ci_successful_runs'] / aggregated['ci_total_runs']

            # Retry analysis
            retry_counts = pr_ci_data.groupby('ci_name').size()
            aggregated['ci_avg_retries'] = retry_counts.mean()
            aggregated['ci_max_retries'] = retry_counts.max()

            # Test type diversity
            test_types = pr_ci_data['ci_name'].unique()
            aggregated['ci_has_build'] = any('build' in name.lower() for name in test_types)
            aggregated['ci_has_test'] = any('test' in name.lower() for name in test_types)
            aggregated['ci_has_lint'] = any(any(keyword in name.lower() for keyword in ['lint', 'style', 'format']) for name in test_types)
            aggregated['ci_has_deploy'] = any('deploy' in name.lower() for name in test_types)

        else:
            # No CI runs with duration data
            aggregated.update({
                'ci_total_duration': 0,
                'ci_avg_duration': 0,
                'ci_max_duration': 0,
                'ci_min_duration': 0,
                'ci_total_runs': len(pr_ci_data),
                'ci_unique_tests': pr_ci_data['ci_name'].nunique(),
                'ci_successful_runs': 0,
                'ci_failed_runs': 0,
                'ci_success_rate': 0,
                'ci_avg_retries': 0,
                'ci_max_retries': 0,
                'ci_has_build': False,
                'ci_has_test': False,
                'ci_has_lint': False,
                'ci_has_deploy': False,
            })

        return aggregated

    def _extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic PR features."""
        features = pd.DataFrame()

        # Numeric features
        numeric_columns = [
            'pr_files_changed', 'pr_additions', 'pr_deletions', 'pr_commits'
        ]

        for col in numeric_columns:
            if col in df.columns:
                features[col] = df[col].fillna(0)

        # Boolean features
        if 'pr_is_draft' in df.columns:
            features['pr_is_draft'] = df['pr_is_draft'].astype(int)

        return features

    def _extract_ci_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract CI-specific features."""
        features = pd.DataFrame()

        # CI numeric features
        ci_numeric_columns = [
            'ci_total_duration', 'ci_avg_duration', 'ci_max_duration', 'ci_min_duration',
            'ci_total_runs', 'ci_unique_tests', 'ci_successful_runs', 'ci_failed_runs',
            'ci_success_rate', 'ci_avg_retries', 'ci_max_retries'
        ]

        for col in ci_numeric_columns:
            if col in df.columns:
                features[col] = df[col].fillna(0)

        # CI boolean features
        ci_boolean_columns = [
            'ci_has_build', 'ci_has_test', 'ci_has_lint', 'ci_has_deploy'
        ]

        for col in ci_boolean_columns:
            if col in df.columns:
                features[col] = df[col].astype(int)

        # Derived features
        if 'ci_total_runs' in features.columns and 'ci_unique_tests' in features.columns:
            features['ci_run_per_test_ratio'] = features['ci_total_runs'] / (features['ci_unique_tests'] + 1)

        if 'pr_files_changed' in df.columns and 'ci_avg_duration' in features.columns:
            features['ci_duration_per_file'] = features['ci_avg_duration'] / (df['pr_files_changed'] + 1)

        return features

    def _extract_text_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract text features from PR titles."""
        if 'pr_title' not in df.columns:
            return None

        text_data = df['pr_title'].fillna('').astype(str).tolist()

        if not any(text_data):  # All empty strings
            return None

        try:
            # Initialize TF-IDF vectorizer if not already done
            if self.tfidf_vectorizer is None:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=self.config.max_ci_text_features,
                    stop_words='english',
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,  # Different from PR model due to potentially less data
                    max_df=0.9,
                    sublinear_tf=True
                )

            # Fit and transform text data
            if not self.is_fitted:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_data)
            else:
                tfidf_matrix = self.tfidf_vectorizer.transform(text_data)

            # Check if transformation was successful
            if tfidf_matrix is None or tfidf_matrix.shape[1] == 0:  # type: ignore
                return None

            # Convert to DataFrame
            feature_names = [f"ci_tfidf_{i}" for i in range(tfidf_matrix.shape[1])]  # type: ignore
            text_features = pd.DataFrame(
                tfidf_matrix.toarray(),  # type: ignore
                columns=feature_names,  # type: ignore
                index=df.index
            )

            return text_features

        except Exception as e:
            logger.warning(f"Error extracting CI text features: {e}")
            return None

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        # For now, we don't have categorical features to encode beyond boolean ones
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        # Fill numeric columns with 0
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)

        return df

    def _balance_feature_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance the influence of text features vs structural features."""
        # Identify text features
        text_cols = [col for col in df.columns if col.startswith('ci_tfidf_')]
        
        if text_cols:
            # Apply weighting factor to reduce text feature influence
            text_weight_factor = self.config.ci_text_feature_weight
            
            df_balanced = df.copy()
            df_balanced[text_cols] = df_balanced[text_cols] * text_weight_factor
            
            logger.info(f"Applied CI feature balancing: {len(text_cols)} text features (weighted by {text_weight_factor})")
            
            return df_balanced
        
        return df

    def _extract_target(self, df: pd.DataFrame, target_type: str) -> Optional[pd.Series]:
        """Extract target variable based on type."""
        if target_type == 'duration':
            # Predict total CI duration
            if 'ci_total_duration' in df.columns:
                # Convert to hours for consistency with PR model
                target = df['ci_total_duration'] / 3600
                return target[target > 0]  # Only positive durations
            
        elif target_type == 'attempts':
            # Predict number of CI attempts needed
            if 'ci_avg_retries' in df.columns:
                target = df['ci_avg_retries']
                return target[target >= 0]  # type: ignore
            
        elif target_type == 'success':
            # Predict CI success rate
            if 'ci_success_rate' in df.columns:
                target = df['ci_success_rate']
                return target[(target >= 0) & (target <= 1)]  # type: ignore
        
        logger.warning(f"Could not extract target for type: {target_type}")
        return None

    def scale_features(self, X_train: pd.DataFrame,
                      X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features using StandardScaler.

        Args:
            X_train: Training features
            X_test: Test features

        Returns:
            Tuple of (scaled_X_train, scaled_X_test)
        """
        # Fit scaler on training data
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        # Transform test data
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        self.is_fitted = True
        return X_train_scaled, X_test_scaled 