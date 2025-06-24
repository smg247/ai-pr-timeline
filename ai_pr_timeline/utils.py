"""
Utility functions for the AI PR Timeline plugin.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )

def save_results_to_json(results: Dict[str, Any], filename: str) -> None:
    """
    Save results dictionary to JSON file.
    
    Args:
        results: Results dictionary
        filename: Output filename
    """
    # Convert numpy types to native Python types for JSON serialization
    def convert_types(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return obj
    
    # Recursively convert types
    def clean_dict(d):
        if isinstance(d, dict):
            return {k: clean_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [clean_dict(item) for item in d]
        else:
            return convert_types(d)
    
    cleaned_results = clean_dict(results)
    
    with open(filename, 'w') as f:
        json.dump(cleaned_results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {filename}")

def load_results_from_json(filename: str) -> Dict[str, Any]:
    """
    Load results dictionary from JSON file.
    
    Args:
        filename: Input filename
        
    Returns:
        Results dictionary
    """
    with open(filename, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Results loaded from {filename}")
    return results

def format_duration(hours: float) -> str:
    """
    Format duration in hours to human-readable string.
    
    Args:
        hours: Duration in hours
        
    Returns:
        Formatted string
    """
    if hours < 1:
        minutes = int(hours * 60)
        return f"{minutes} minutes"
    elif hours < 24:
        return f"{hours:.1f} hours"
    else:
        days = hours / 24
        if days < 7:
            return f"{days:.1f} days"
        else:
            weeks = days / 7
            return f"{weeks:.1f} weeks"

def calculate_business_hours(start_time: datetime, end_time: datetime) -> float:
    """
    Calculate business hours between two timestamps.
    Assumes business hours are 9 AM - 5 PM, Monday - Friday.
    
    Args:
        start_time: Start timestamp
        end_time: End timestamp
        
    Returns:
        Business hours as float
    """
    business_hours = 0.0
    current_time = start_time
    
    while current_time < end_time:
        # Check if current day is a weekday
        if current_time.weekday() < 5:  # Monday = 0, Friday = 4
            # Get start and end of business day
            business_start = current_time.replace(hour=9, minute=0, second=0, microsecond=0)
            business_end = current_time.replace(hour=17, minute=0, second=0, microsecond=0)
            
            # Calculate overlap with our time range
            overlap_start = max(current_time, business_start)
            overlap_end = min(end_time, business_end)
            
            if overlap_start < overlap_end:
                business_hours += (overlap_end - overlap_start).total_seconds() / 3600
        
        # Move to next day
        current_time = (current_time + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    return business_hours

def get_pr_size_category(additions: int, deletions: int) -> str:
    """
    Categorize PR size based on lines changed.
    
    Args:
        additions: Number of lines added
        deletions: Number of lines deleted
        
    Returns:
        Size category string
    """
    total_changes = additions + deletions
    
    if total_changes <= 10:
        return "XS"
    elif total_changes <= 50:
        return "S"
    elif total_changes <= 200:
        return "M"
    elif total_changes <= 500:
        return "L"
    else:
        return "XL"

def validate_pr_data(pr_data: Dict[str, Any]) -> List[str]:
    """
    Validate PR data dictionary for required fields.
    
    Args:
        pr_data: PR data dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    required_fields = [
        'title', 'body', 'review_count', 'comment_count', 'commit_count',
        'files_changed', 'additions', 'deletions', 'created_hour', 'created_day',
        'author_association', 'is_draft'
    ]
    
    errors = []
    
    for field in required_fields:
        if field not in pr_data:
            errors.append(f"Missing required field: {field}")
        elif pr_data[field] is None:
            errors.append(f"Field cannot be None: {field}")
    
    # Validate data types
    if 'review_count' in pr_data and not isinstance(pr_data['review_count'], (int, float)):
        errors.append("review_count must be numeric")
    
    if 'created_hour' in pr_data and not (0 <= pr_data['created_hour'] <= 23):
        errors.append("created_hour must be between 0 and 23")
    
    if 'created_day' in pr_data and not (0 <= pr_data['created_day'] <= 6):
        errors.append("created_day must be between 0 and 6")
    
    return errors

def create_pr_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create summary statistics for PR data.
    
    Args:
        df: DataFrame with PR data
        
    Returns:
        Dictionary with summary statistics
    """
    if df.empty:
        return {"error": "No data provided"}
    
    stats = {
        "total_prs": len(df),
        "merge_time_stats": {
            "mean_hours": df['merge_time_hours'].mean(),
            "median_hours": df['merge_time_hours'].median(),
            "std_hours": df['merge_time_hours'].std(),
            "min_hours": df['merge_time_hours'].min(),
            "max_hours": df['merge_time_hours'].max(),
            "percentiles": {
                "25th": df['merge_time_hours'].quantile(0.25),
                "75th": df['merge_time_hours'].quantile(0.75),
                "90th": df['merge_time_hours'].quantile(0.90),
                "95th": df['merge_time_hours'].quantile(0.95)
            }
        },
        "size_distribution": {
            "mean_files_changed": df['files_changed'].mean(),
            "mean_additions": df['additions'].mean(),
            "mean_deletions": df['deletions'].mean(),
            "mean_total_changes": (df['additions'] + df['deletions']).mean()
        },
        "activity_stats": {
            "mean_reviews": df['review_count'].mean(),
            "mean_comments": df['comment_count'].mean(),
            "mean_commits": df['commit_count'].mean()
        },
        "timing_patterns": {
            "weekend_prs": len(df[df['created_day'].isin([5, 6])]),
            "business_hours_prs": len(df[(df['created_hour'] >= 9) & (df['created_hour'] <= 17)]),
            "draft_prs": len(df[df['is_draft'] == True])
        }
    }
    
    # Add percentages
    total = len(df)
    stats["timing_patterns"]["weekend_percentage"] = (stats["timing_patterns"]["weekend_prs"] / total) * 100
    stats["timing_patterns"]["business_hours_percentage"] = (stats["timing_patterns"]["business_hours_prs"] / total) * 100
    stats["timing_patterns"]["draft_percentage"] = (stats["timing_patterns"]["draft_prs"] / total) * 100
    
    return stats 