"""
Feature engineering module for processing PR data.
"""

import logging
import re
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Processes and engineers features from raw PR data."""
    
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.text_vectorizer = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Engineer features from raw PR data.
        
        Args:
            df: Raw PR data DataFrame
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info("Engineering features from PR data")
        
        # Create a copy to avoid modifying original data
        features_df = df.copy()
        
        # Remove outliers (PRs that took more than 30 days) - only for training data
        if 'merge_time_hours' in features_df.columns:
            features_df = features_df[features_df['merge_time_hours'] <= 24 * 30]
        
        # Basic numerical features
        numerical_features = [
            'review_count', 'comment_count', 'commit_count',
            'files_changed', 'additions', 'deletions',
            'created_hour', 'created_day'
        ]
        
        # Create derived features
        features_df['total_changes'] = features_df['additions'] + features_df['deletions']
        features_df['change_ratio'] = np.where(
            features_df['deletions'] > 0,
            features_df['additions'] / features_df['deletions'],
            features_df['additions']
        )
        features_df['files_per_commit'] = np.where(
            features_df['commit_count'] > 0,
            features_df['files_changed'] / features_df['commit_count'],
            0
        )
        features_df['changes_per_file'] = np.where(
            features_df['files_changed'] > 0,
            features_df['total_changes'] / features_df['files_changed'],
            0
        )
        
        # Time-based features
        features_df['is_weekend'] = (features_df['created_day'] == 5) | (features_df['created_day'] == 6)  # Saturday, Sunday
        features_df['is_business_hours'] = (
            (features_df['created_hour'] >= 9) & 
            (features_df['created_hour'] <= 17)
        )
        
        # Categorical features
        features_df['author_is_member'] = (
            (features_df['author_association'] == 'MEMBER') |
            (features_df['author_association'] == 'OWNER') |
            (features_df['author_association'] == 'COLLABORATOR')
        )
        
        # Text features
        if self.config.include_text_features:
            text_features = self._extract_text_features(features_df)
            features_df = pd.concat([features_df, text_features], axis=1)
        
        # Select final features
        feature_columns = numerical_features + [
            'total_changes', 'change_ratio', 'files_per_commit', 'changes_per_file',
            'is_weekend', 'is_business_hours', 'author_is_member', 'is_draft'
        ]
        
        if self.config.include_text_features and hasattr(self, '_text_feature_names'):
            feature_columns.extend(self._text_feature_names)
        
        # Handle missing values
        features_df = features_df.fillna(0)
        
        # Extract target variable (only exists for training data)
        target = features_df['merge_time_hours'] if 'merge_time_hours' in features_df.columns else None
        
        # Select and return features
        final_features = features_df[feature_columns]
        
        logger.info(f"Engineered {len(feature_columns)} features")
        return final_features, target
    
    def _extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from PR title and body text.
        
        Args:
            df: DataFrame with title and body columns
            
        Returns:
            DataFrame with text features
        """
        logger.info("Extracting text features")
        
        # Combine title and body
        text_data = (df['title'].fillna('') + ' ' + df['body'].fillna('')).str.lower()
        
        # Clean text
        text_data = text_data.apply(self._clean_text)
        
        # Initialize or fit vectorizer
        if self.text_vectorizer is None:
            self.text_vectorizer = TfidfVectorizer(
                max_features=self.config.max_text_features,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            text_matrix = self.text_vectorizer.fit_transform(text_data)
        else:
            text_matrix = self.text_vectorizer.transform(text_data)
        
        # Create feature names
        feature_names = [f'text_{i}' for i in range(text_matrix.shape[1])]
        self._text_feature_names = feature_names
        
        # Convert to DataFrame
        text_df = pd.DataFrame(
            text_matrix.toarray(),
            columns=feature_names,
            index=df.index
        )
        
        return text_df
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text data."""
        if not isinstance(text, str):
            return ''
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`.*?`', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale numerical features.
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            
        Returns:
            Tuple of scaled training and test features
        """
        logger.info("Scaling features")
        
        # Identify numerical columns (excluding binary features)
        numerical_cols = [
            col for col in X_train.columns 
            if col not in ['is_weekend', 'is_business_hours', 'author_is_member', 'is_draft']
            and not col.startswith('text_')
        ]
        
        # Fit scaler on training data
        X_train_scaled = X_train.copy()
        X_train_scaled[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled, None
    
    def create_time_bins(self, target: pd.Series, n_bins: int = 5) -> pd.Series:
        """
        Convert continuous target to time bins for classification.
        
        Args:
            target: Continuous merge time target
            n_bins: Number of bins to create
            
        Returns:
            Binned target labels
        """
        # Create bins based on quantiles
        bin_edges = np.quantile(target, np.linspace(0, 1, n_bins + 1))
        bin_labels = [f'bin_{i}' for i in range(n_bins)]
        
        binned_target = pd.cut(target, bins=bin_edges, labels=bin_labels, include_lowest=True)
        
        logger.info(f"Created {n_bins} time bins")
        return binned_target 