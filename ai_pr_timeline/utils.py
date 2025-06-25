"""
Utility functions for the AI PR Timeline plugin.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def setup_logging(level: str = "INFO") -> None:
    """
    Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration parameters.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, raises ValueError if invalid
    """
    required_fields = ['github_token', 'model_type']

    for field in required_fields:
        if field not in config or config[field] is None:
            raise ValueError(f"Missing required configuration field: {field}")

    valid_model_types = ['random_forest', 'xgboost', 'lightgbm']
    if config['model_type'] not in valid_model_types:
        raise ValueError(f"Invalid model_type. Must be one of: {valid_model_types}")

    if 'test_size' in config:
        if not 0 < config['test_size'] < 1:
            raise ValueError("test_size must be between 0 and 1")

    return True


def create_directories(paths: List[str]) -> None:
    """
    Create directories if they don't exist.

    Args:
        paths: List of directory paths to create
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save data to JSON file.

    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary with loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def format_duration(hours: float) -> str:
    """
    Format duration in hours to human-readable string.

    Args:
        hours: Duration in hours

    Returns:
        Formatted duration string
    """
    if hours < 1:
        minutes = int(hours * 60)
        return f"{minutes} minutes"
    elif hours < 24:
        return f"{hours:.1f} hours"
    else:
        days = hours / 24
        return f"{days:.1f} days"


def calculate_percentiles(data: List[float], percentiles: Optional[List[int]] = None) -> Dict[str, float]:
    """
    Calculate percentiles for a list of values.

    Args:
        data: List of numeric values
        percentiles: List of percentile values to calculate (default: [25, 50, 75, 90, 95])

    Returns:
        Dictionary with percentile values
    """
    if percentiles is None:
        percentiles = [25, 50, 75, 90, 95]

    if not data:
        return {f"p{p}": 0.0 for p in percentiles}

    result = {}
    for p in percentiles:
        result[f"p{p}"] = np.percentile(data, p)

    return result


def filter_outliers(data: pd.Series, method: str = 'iqr', factor: float = 1.5) -> pd.Series:
    """
    Filter outliers from data series.

    Args:
        data: Pandas Series with numeric data
        method: Method to use ('iqr' or 'zscore')
        factor: Factor for outlier detection

    Returns:
        Filtered Series without outliers
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        return data[(data >= lower_bound) & (data <= upper_bound)]  # type: ignore
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        return data[z_scores < factor]  # type: ignore
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")


def get_business_hours_only(timestamp: datetime) -> bool:
    """
    Check if timestamp falls within business hours (9 AM - 5 PM, Monday-Friday).

    Args:
        timestamp: Datetime object to check

    Returns:
        True if within business hours, False otherwise
    """
    # Monday = 0, Sunday = 6
    if timestamp.weekday() >= 5:  # Weekend
        return False

    hour = timestamp.hour
    return 9 <= hour <= 17


def calculate_business_hours_between(start: datetime, end: datetime) -> float:
    """
    Calculate business hours between two timestamps.

    Args:
        start: Start timestamp
        end: End timestamp

    Returns:
        Number of business hours between timestamps
    """
    if start >= end:
        return 0.0

    business_hours = 0.0
    current = start

    while current < end:
        # Check if current time is in business hours
        if get_business_hours_only(current):
            # Calculate hours until end of business day or end time
            end_of_business = current.replace(hour=17, minute=0, second=0, microsecond=0)
            if end_of_business > end:
                end_of_business = end

            # Add business hours for this period
            if current < end_of_business:
                hours_diff = (end_of_business - current).total_seconds() / 3600
                business_hours += hours_diff

        # Move to next business day start
        next_day = current.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
        while next_day.weekday() >= 5:  # Skip weekends
            next_day += timedelta(days=1)

        current = next_day

    return business_hours


def validate_pr_data(pr_data: Dict[str, Any]) -> bool:
    """
    Validate PR data dictionary.

    Args:
        pr_data: PR data dictionary

    Returns:
        True if valid, raises ValueError if invalid
    """
    required_fields = ['pr_number', 'title', 'review_count', 'files_changed']

    for field in required_fields:
        if field not in pr_data:
            raise ValueError(f"Missing required PR data field: {field}")

    # Validate numeric fields
    numeric_fields = ['review_count', 'comment_count', 'commit_count',
                     'files_changed', 'additions', 'deletions']

    for field in numeric_fields:
        if field in pr_data:
            value = pr_data[field]
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"Field {field} must be a non-negative number")

    return True


def aggregate_repository_stats(pr_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate statistics from multiple PR data entries.

    Args:
        pr_data_list: List of PR data dictionaries

    Returns:
        Dictionary with aggregated statistics
    """
    if not pr_data_list:
        return {}

    # Extract numeric fields
    numeric_fields = ['review_count', 'comment_count', 'commit_count',
                     'files_changed', 'additions', 'deletions']

    stats = {}

    for field in numeric_fields:
        values = [pr.get(field, 0) for pr in pr_data_list if pr.get(field) is not None]
        if values:
            stats[field] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

    # Calculate merge time stats if available
    merge_times = [pr.get('merge_time_hours') for pr in pr_data_list
                  if pr.get('merge_time_hours') is not None]

    if merge_times:
        stats['merge_time_hours'] = {
            'mean': np.mean(merge_times),  # type: ignore
            'median': np.median(merge_times),  # type: ignore
            'std': np.std(merge_times),  # type: ignore
            'min': np.min(merge_times),  # type: ignore
            'max': np.max(merge_times),  # type: ignore
            'percentiles': calculate_percentiles(merge_times)  # type: ignore
        }

    # Repository-level stats
    stats['total_prs'] = len(pr_data_list)
    repositories = set(pr.get('repository') for pr in pr_data_list if pr.get('repository'))
    stats['unique_repositories'] = len(repositories)

    # Time-based analysis
    weekend_prs = sum(1 for pr in pr_data_list if pr.get('created_day') in [5, 6])
    stats['weekend_prs_ratio'] = weekend_prs / len(pr_data_list) if pr_data_list else 0

    business_hour_prs = sum(1 for pr in pr_data_list
                           if pr.get('created_hour') is not None and
                           9 <= pr.get('created_hour', 0) <= 17)
    stats['business_hours_prs_ratio'] = (business_hour_prs / len(pr_data_list)
                                        if pr_data_list else 0)

    return stats
