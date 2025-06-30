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

    def engineer_features(self, pr_data_list: List[Dict],
                         target_type: str = 'duration',
                         include_target: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Engineer features from PR data with embedded CI data.

        Args:
            pr_data_list: List of PR data dictionaries with embedded ci_data
            target_type: Type of target to predict ('duration', 'attempts', 'success')
            include_target: Whether to include target variable

        Returns:
            Tuple of (features_df, target_series) or (features_df, None)
        """
        logger.info(f"Engineering CI features for target: {target_type}")

        if not pr_data_list:
            return pd.DataFrame(), None

        pr_aggregated = self._aggregate_ci_from_pr_data(pr_data_list)

        if not pr_aggregated:
            return pd.DataFrame(), None

        df = pd.DataFrame(pr_aggregated)

        features = self._extract_basic_features(df)

        ci_features = self._extract_ci_features(df)
        features = pd.concat([features, ci_features], axis=1)

        if self.config.include_ci_text_features:
            text_features = self._extract_text_features(df)
            if text_features is not None and not text_features.empty:
                features = pd.concat([features, text_features], axis=1)

        features = self._encode_categorical_features(features)
        features = self._handle_missing_values(features)
        features = self._balance_feature_weights(features)

        logger.info(f"Engineered {len(features.columns)} CI features")

        target = None
        if include_target:
            target = self._extract_target(df, target_type)

        return features, target

    def _aggregate_ci_from_pr_data(self, pr_data_list: List[Dict]) -> List[Dict]:
        """
        Aggregate CI data from PR data list with embedded ci_data.

        Args:
            pr_data_list: List of PR data dictionaries with embedded ci_data

        Returns:
            List of aggregated PR data with CI features
        """
        aggregated_data = []

        for pr_data in pr_data_list:
            ci_data = pr_data.get('ci_data', [])
            
            if not ci_data:
                continue
                
            aggregated_pr = self._aggregate_single_pr_normalized(pr_data, ci_data)
            if aggregated_pr:
                aggregated_data.append(aggregated_pr)

        return aggregated_data

    def _aggregate_single_pr_normalized(self, pr_data: Dict, ci_data: List[Dict]) -> Dict:
        """Aggregate CI data for a single PR with normalized format."""
        if not ci_data:
            return {}

        aggregated = {
            'pr_title': pr_data.get('title', ''),
            'pr_number': pr_data.get('pr_number'),
            'repository': pr_data.get('repository'),
            'pr_files_changed': pr_data.get('files_changed', 0),
            'pr_additions': pr_data.get('additions', 0),
            'pr_deletions': pr_data.get('deletions', 0),
            'pr_commits': pr_data.get('commit_count', 0),
            'pr_is_draft': pr_data.get('is_draft', False),
        }

        ci_runs_with_duration = [run for run in ci_data if run.get('ci_duration_seconds') is not None]
        
        if ci_runs_with_duration:
            durations = [run['ci_duration_seconds'] for run in ci_runs_with_duration]
            
            aggregated['ci_total_duration'] = sum(durations)
            aggregated['ci_avg_duration'] = sum(durations) / len(durations)
            aggregated['ci_max_duration'] = max(durations)
            aggregated['ci_min_duration'] = min(durations)
        else:
            aggregated.update({
                'ci_total_duration': 0,
                'ci_avg_duration': 0,
                'ci_max_duration': 0,
                'ci_min_duration': 0,
            })

        aggregated['ci_total_runs'] = len(ci_data)
        aggregated['ci_unique_tests'] = len(set(run.get('ci_name', '') for run in ci_data))
        aggregated['ci_successful_runs'] = len([run for run in ci_data if run.get('ci_state') == 'success'])
        aggregated['ci_failed_runs'] = len([run for run in ci_data if run.get('ci_state') in ['failure', 'error']])
        aggregated['ci_success_rate'] = aggregated['ci_successful_runs'] / aggregated['ci_total_runs'] if aggregated['ci_total_runs'] > 0 else 0

        test_counts = {}
        for run in ci_data:
            test_name = run.get('ci_name', '')
            test_counts[test_name] = test_counts.get(test_name, 0) + 1
        
        if test_counts:
            retry_counts = list(test_counts.values())
            aggregated['ci_avg_retries'] = sum(retry_counts) / len(retry_counts)
            aggregated['ci_max_retries'] = max(retry_counts)
        else:
            aggregated['ci_avg_retries'] = 0
            aggregated['ci_max_retries'] = 0

        test_names = [run.get('ci_name', '').lower() for run in ci_data]
        aggregated['ci_has_build'] = any('build' in name for name in test_names)
        aggregated['ci_has_test'] = any('test' in name for name in test_names)
        aggregated['ci_has_lint'] = any(any(keyword in name for keyword in ['lint', 'style', 'format']) for name in test_names)
        aggregated['ci_has_deploy'] = any('deploy' in name for name in test_names)

        return aggregated

    def _extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic PR features."""
        features = pd.DataFrame()

        numeric_columns = [
            'pr_files_changed', 'pr_additions', 'pr_deletions', 'pr_commits'
        ]

        for col in numeric_columns:
            if col in df.columns:
                features[col] = df[col].fillna(0)

        if 'pr_is_draft' in df.columns:
            features['pr_is_draft'] = df['pr_is_draft'].fillna(False).astype(int)

        return features

    def _extract_ci_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract CI-specific features."""
        features = pd.DataFrame()

        ci_numeric_columns = [
            'ci_total_duration', 'ci_avg_duration', 'ci_max_duration', 'ci_min_duration',
            'ci_total_runs', 'ci_unique_tests', 'ci_successful_runs', 'ci_failed_runs',
            'ci_success_rate', 'ci_avg_retries', 'ci_max_retries'
        ]

        for col in ci_numeric_columns:
            if col in df.columns:
                features[col] = df[col].fillna(0)

        ci_boolean_columns = [
            'ci_has_build', 'ci_has_test', 'ci_has_lint', 'ci_has_deploy'
        ]

        for col in ci_boolean_columns:
            if col in df.columns:
                features[col] = df[col].astype(int)

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

        if not any(text_data):
            return None

        try:
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

            if not self.is_fitted:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_data)
            else:
                tfidf_matrix = self.tfidf_vectorizer.transform(text_data)

            if tfidf_matrix is None or tfidf_matrix.shape[1] == 0:  # type: ignore
                return None

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
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)

        return df

    def _balance_feature_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance the influence of text features vs structural features."""
        text_cols = [col for col in df.columns if col.startswith('ci_tfidf_')]
        
        if text_cols:
            text_weight_factor = self.config.ci_text_feature_weight
            
            df_balanced = df.copy()
            df_balanced[text_cols] = df_balanced[text_cols] * text_weight_factor
            
            logger.info(f"Applied CI feature balancing: {len(text_cols)} text features (weighted by {text_weight_factor})")
            
            return df_balanced
        
        return df

    def _extract_target(self, df: pd.DataFrame, target_type: str) -> Optional[pd.Series]:
        """Extract target variable based on type."""
        if target_type == 'duration':
            if 'ci_total_duration' in df.columns:
                target = df['ci_total_duration'] / 3600  # Convert to hours for consistency with PR model
                return target[target > 0]
            
        elif target_type == 'attempts':
            if 'ci_avg_retries' in df.columns:
                target = df['ci_avg_retries']
                return target[target >= 0]  # type: ignore
            
        elif target_type == 'success':
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
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        self.is_fitted = True
        return X_train_scaled, X_test_scaled 