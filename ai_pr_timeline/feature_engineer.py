"""
Feature engineering module for PR timeline prediction.
"""

import logging
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature engineering for PR data."""

    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.tfidf_vectorizer = None
        self.is_fitted = False

    def engineer_features(self, df: pd.DataFrame,
                         include_target: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Engineer features from raw PR data.

        Args:
            df: Raw PR data DataFrame
            include_target: Whether to include target variable (merge_time_hours)

        Returns:
            Tuple of (features_df, target_series) or (features_df, None)
        """
        logger.info("Engineering features from PR data")

        # Make a copy to avoid modifying original data
        data = df.copy()

        # Basic features
        features = self._extract_basic_features(data)

        # Derived features
        derived_features = self._extract_derived_features(data)
        features = pd.concat([features, derived_features], axis=1)

        # Text features (optional)
        if self.config.include_text_features:
            text_features = self._extract_text_features(data)
            if text_features is not None and not text_features.empty:
                features = pd.concat([features, text_features], axis=1)

        # Handle categorical variables
        features = self._encode_categorical_features(features)

        # Handle missing values
        features = self._handle_missing_values(features)

        # Balance feature importance between text and structural features
        features = self._balance_feature_weights(features)

        # Ensure features is a DataFrame
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        # Log feature distribution for transparency
        self._log_feature_distribution(features)
        
        logger.info(f"Engineered {len(features.columns)} features")

        # Extract target variable if requested and available
        target = None
        if include_target and 'merge_time_hours' in data.columns:
            target = data['merge_time_hours'].copy()
            # Remove outliers (PRs that took more than 30 days)
            outlier_mask = target <= (30 * 24)  # 30 days in hours
            features = features[outlier_mask]
            target = target[outlier_mask]
            logger.info(f"Removed {(~outlier_mask).sum()} outliers "
                       f"(PRs taking more than 30 days)")

        return features, target  # type: ignore

    def _extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic features from PR data."""
        features = pd.DataFrame()

        # Numeric features
        numeric_columns = [
            'review_count', 'comment_count', 'commit_count',
            'files_changed', 'additions', 'deletions'
        ]

        for col in numeric_columns:
            if col in df.columns:
                features[col] = df[col].fillna(0)

        # Time-based features
        if 'created_hour' in df.columns:
            features['created_hour'] = df['created_hour']

        if 'created_day' in df.columns:
            features['created_day'] = df['created_day']

        # Boolean features
        if 'is_draft' in df.columns:
            features['is_draft'] = df['is_draft'].astype(int)

        return features

    def _extract_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract derived features from PR data."""
        features = pd.DataFrame()

        # Code change ratios
        if all(col in df.columns for col in ['additions', 'deletions', 'files_changed']):
            features['lines_changed'] = df['additions'] + df['deletions']
            features['addition_ratio'] = df['additions'] / (df['additions'] + df['deletions'] + 1)
            features['files_per_addition'] = df['files_changed'] / (df['additions'] + 1)

        # Activity ratios
        if all(col in df.columns for col in ['review_count', 'comment_count', 'commit_count']):
            features['reviews_per_commit'] = df['review_count'] / (df['commit_count'] + 1)
            features['comments_per_commit'] = df['comment_count'] / (df['commit_count'] + 1)

        # Time-based derived features
        if 'created_hour' in df.columns:
            # Business hours (9 AM to 5 PM)
            features['is_business_hours'] = ((df['created_hour'] >= 9) &
                                           (df['created_hour'] <= 17)).astype(int)

        if 'created_day' in df.columns:
            # Weekend (Saturday=5, Sunday=6)
            features['is_weekend'] = (df['created_day'].isin([5, 6])).astype(int)

        return features

    def _extract_text_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract text features using TF-IDF."""
        if 'title' not in df.columns and 'body' not in df.columns:
            return None

        # Combine title and body
        text_data = []
        for _, row in df.iterrows():
            title = str(row.get('title', '')) if 'title' in df.columns else ''
            body = str(row.get('body', '')) if 'body' in df.columns else ''
            combined_text = f"{title} {body}".strip()
            text_data.append(combined_text)

        if not any(text_data):  # All empty strings
            return None

        try:
            # Initialize TF-IDF vectorizer if not already done
            if self.tfidf_vectorizer is None:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=self.config.max_text_features,
                    stop_words='english',
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,  # Ignore terms that appear in fewer than 2 documents
                    max_df=0.8,  # Ignore terms that appear in more than 80% of documents
                    sublinear_tf=True  # Apply sublinear tf scaling to reduce impact of high-frequency terms
                )

            # Fit and transform text data
            if not self.is_fitted:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_data)
            else:
                tfidf_matrix = self.tfidf_vectorizer.transform(text_data)

            # Convert to DataFrame
            feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]  # type: ignore
            text_features = pd.DataFrame(
                tfidf_matrix.toarray(),  # type: ignore
                columns=feature_names,  # type: ignore
                index=df.index
            )

            return text_features

        except Exception as e:
            logger.warning(f"Error extracting text features: {e}")
            return None

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        # For now, we don't have categorical features to encode
        # This method is here for future extensions
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        # Fill numeric columns with 0
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)

        return df

    def _balance_feature_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance the influence of text features vs structural features.
        
        Text features are reduced in magnitude to prevent them from dominating
        the model due to their large number (up to max_text_features).
        
        Args:
            df: DataFrame with all features
            
        Returns:
            DataFrame with balanced feature weights
        """
        # Identify text features (TF-IDF features)
        text_cols = [col for col in df.columns if col.startswith('tfidf_')]
        
        if text_cols:
            # Apply weighting factor to reduce text feature influence
            # This helps structural features (like files_changed, review_count) 
            # maintain their importance relative to text features
            text_weight_factor = self.config.text_feature_weight
            
            df_balanced = df.copy()
            df_balanced[text_cols] = df_balanced[text_cols] * text_weight_factor
            
            logger.info(f"Applied feature balancing: {len(text_cols)} text features (weighted by {text_weight_factor})")
            
            return df_balanced
        
        return df

    def _log_feature_distribution(self, df: pd.DataFrame) -> None:
        """Log the distribution of feature types for transparency."""
        text_cols = [col for col in df.columns if col.startswith('tfidf_')]
        structural_cols = [col for col in df.columns if not col.startswith('tfidf_')]
        
        logger.info(f"Feature distribution: {len(structural_cols)} structural, "
                   f"{len(text_cols)} text features")
        
        if text_cols and structural_cols:
            # Calculate ratio for transparency
            ratio = len(text_cols) / len(structural_cols)
            logger.info(f"Text-to-structural feature ratio: {ratio:.1f}:1 "
                       f"(weighted by {self.config.text_feature_weight})")

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
