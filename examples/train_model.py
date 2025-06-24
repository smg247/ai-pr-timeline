#!/usr/bin/env python3
"""
Example script for training a PR timeline prediction model.

Usage:
    python examples/train_model.py --repo "owner/repo" --token "your_github_token"
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add parent directory to path to import our module
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_pr_timeline import PRTimelinePredictor, Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    parser = argparse.ArgumentParser(description='Train PR timeline prediction model')
    parser.add_argument('--repo', required=True, help='Repository name (owner/repo)')
    parser.add_argument('--token', help='GitHub token (or set GITHUB_TOKEN env var)')
    parser.add_argument('--model-type', default='random_forest', 
                       choices=['random_forest', 'xgboost', 'lightgbm'],
                       help='Type of model to train')
    parser.add_argument('--tune-hyperparams', action='store_true',
                       help='Perform hyperparameter tuning')
    parser.add_argument('--max-prs', type=int, default=1000,
                       help='Maximum number of PRs to collect')
    
    args = parser.parse_args()
    
    # Set up configuration
    config = Config()
    config.github_token = args.token or os.getenv('GITHUB_TOKEN')
    config.model_type = args.model_type
    config.max_prs_per_repo = args.max_prs
    
    if not config.github_token:
        print("Error: GitHub token is required. Set GITHUB_TOKEN environment variable or use --token")
        sys.exit(1)
    
    try:
        # Initialize predictor
        predictor = PRTimelinePredictor(config)
        
        print(f"Training model on repository: {args.repo}")
        print(f"Model type: {args.model_type}")
        print(f"Max PRs to collect: {args.max_prs}")
        print("-" * 50)
        
        # Train the model
        results = predictor.train_on_repository(
            args.repo,
            save_model=True
        )
        
        print("\nTraining completed!")
        print(f"Data points used: {results['data_points']}")
        print(f"Training set size: {results['training_size']}")
        print(f"Test set size: {results['test_size']}")
        
        print("\nModel Performance:")
        metrics = results['metrics']
        print(f"  MAE (Mean Absolute Error): {metrics['mae']:.2f} hours")
        print(f"  RMSE (Root Mean Squared Error): {metrics['rmse']:.2f} hours")
        print(f"  R² Score: {metrics['r2']:.3f}")
        print(f"  MAPE (Mean Absolute Percentage Error): {metrics['mape']:.1f}%")
        
        print("\nCross-Validation Results:")
        cv_metrics = results['cv_metrics']
        print(f"  CV MAE: {cv_metrics['cv_mae_mean']:.2f} ± {cv_metrics['cv_mae_std']:.2f} hours")
        
        print("\nTop 10 Most Important Features:")
        for i, feature in enumerate(results['feature_importance'][:10], 1):
            print(f"  {i:2d}. {feature['feature']}: {feature['importance']:.4f}")
        
        print("\nModel saved successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 