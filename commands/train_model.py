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
from datetime import datetime

# Add parent directory to path to import our module
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_pr_timeline import PRTimelinePredictor, Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    parser = argparse.ArgumentParser(description='Train PR timeline prediction model using cached data')
    parser.add_argument('--repos', required=True, nargs='+',
                       help='Repository names in format owner/repo (uses cached data)')

    parser.add_argument('--model-type', default='random_forest', 
                       choices=['random_forest', 'xgboost', 'lightgbm'],
                       help='Type of model to train')
    parser.add_argument('--tune-hyperparams', action='store_true',
                       help='Perform hyperparameter tuning')
    parser.add_argument('--text-weight', type=float, default=0.3,
                       help='Weight factor for text features (0.0-1.0, default: 0.3)')
    parser.add_argument('--max-text-features', type=int, default=50,
                       help='Maximum number of text features to extract (default: 50)')
    parser.add_argument('--output-filename', 
                       help='Custom filename for the saved model (default: auto-generated with timestamp)')
    
    args = parser.parse_args()
    
    # Set up configuration (no GitHub token needed for cache-only training)
    config = Config()
    config.github_token = None  # Not needed for training with cached data
    config.model_type = args.model_type
    config.text_feature_weight = args.text_weight
    config.max_text_features = args.max_text_features
    
    # Validate text-weight parameter
    if not 0.0 <= args.text_weight <= 1.0:
        print(f"Error: --text-weight ({args.text_weight}) must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Validate max-text-features parameter
    if args.max_text_features < 1:
        print(f"Error: --max-text-features ({args.max_text_features}) must be at least 1")
        sys.exit(1)
    
    try:
        # Initialize predictor in cache-only mode
        predictor = PRTimelinePredictor(config, cache_only=True)
        
        print(f"Training model on repositories: {', '.join(args.repos)}")
        print(f"Model type: {args.model_type}")
        print(f"Text feature settings: max={args.max_text_features}, weight={args.text_weight}")
        print(f"Using cached data only (no API calls)")
        print("-" * 50)
        
        # Train the model using cached data only
        results = predictor.train_on_cached_data(
            repo_names=args.repos,
            model_type=args.model_type,
            hyperparameter_tuning=args.tune_hyperparams
        )
        
        # Save the trained model
        if args.output_filename:
            model_filename = args.output_filename
        else:
            # Auto-generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            repos_str = "_".join(repo.replace('/', '-') for repo in args.repos)
            model_filename = f"{repos_str}_{args.model_type}_model_{timestamp}.pkl"
        
        predictor.save_model(model_filename)
        
        print("\nTraining completed!")
        print(f"Data points used: {results['data_points']}")
        print(f"Training set size: {results['training_size']}")
        print(f"Test set size: {results['test_size']}")
        print(f"Repositories: {', '.join(args.repos)}")
        
        print("\nModel Performance:")
        metrics = results['metrics']
        print(f"  MAE (Mean Absolute Error): {metrics['mae']:.2f} hours")
        print(f"  RMSE (Root Mean Squared Error): {metrics['rmse']:.2f} hours")
        print(f"  R² Score: {metrics['r2']:.3f}")
        print(f"  MAPE (Mean Absolute Percentage Error): {metrics['mape']:.1f}%")
        
        print("\nTop 10 Most Important Features:")
        for i, feature in enumerate(results['feature_importance'][:10], 1):
            print(f"  {i:2d}. {feature['feature']}: {feature['importance']:.4f}")
        
        print(f"\nModel saved successfully as: {model_filename}")
        print(f"\n💡 Tip: Use collect_data.py to refresh cached data when needed")
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 