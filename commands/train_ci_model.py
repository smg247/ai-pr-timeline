#!/usr/bin/env python3
"""
Command-line tool for training CI prediction models.
"""

import argparse
import sys
import os
import json
from pathlib import Path

# Add the parent directory to Python path to import ai_pr_timeline
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_pr_timeline import CIPredictor, Config
from ai_pr_timeline.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description='Train CI prediction models using cached data')
    parser.add_argument('--repos', required=True, nargs='+',
                       help='Repository names in format owner/repo (uses cached data)')
    parser.add_argument('--model-type', default='random_forest',
                       choices=['random_forest', 'xgboost', 'lightgbm'],
                       help='Type of model to train (default: random_forest)')
    parser.add_argument('--model-name', default='ci_model',
                       help='Base name for saved models (default: ci_model)')
    parser.add_argument('--hyperparameter-tuning', action='store_true',
                       help='Enable hyperparameter tuning (slower but potentially better)')
    parser.add_argument('--output', help='Output file for training results (JSON)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Set up configuration (no GitHub token needed for cache-only training)
    config = Config()
    config.github_token = None  # Not needed for training with cached data
    config.ci_model_type = args.model_type
    
    try:
        # Initialize CI predictor in cache-only mode
        ci_predictor = CIPredictor(config, cache_only=True)
        
        print(f"Training CI models on repositories: {', '.join(args.repos)}")
        print(f"Model type: {args.model_type}")
        print(f"Using cached data only (no API calls)")
        
        if args.hyperparameter_tuning:
            print("Hyperparameter tuning enabled - this may take longer")
        print("-" * 50)
        
        # Train models using cached data only
        results = ci_predictor.train_models_on_cached_data(
            repo_names=args.repos,
            hyperparameter_tuning=args.hyperparameter_tuning
        )
        
        # Save models
        ci_predictor.save_models(args.model_name)
        
        # Display results
        print("\n" + "="*60)
        print("CI MODEL TRAINING RESULTS")
        print("="*60)
        
        for model_type, result in results.items():
            print(f"\n{model_type.upper()} MODEL:")
            if 'error' in result:
                print(f"  Error: {result['error']}")
            else:
                metrics = result['metrics']
                print(f"  MAE: {metrics['mae']:.4f}")
                print(f"  RMSE: {metrics['rmse']:.4f}")
                print(f"  R²: {metrics['r2']:.4f}")
                if 'mae_minutes' in metrics:
                    print(f"  MAE (minutes): {metrics['mae_minutes']:.2f}")
                if 'accuracy' in metrics:
                    print(f"  Accuracy: {metrics['accuracy']:.4f}")
                
                print(f"  Top features:")
                for i, feature in enumerate(result['feature_importance'][:5]):
                    print(f"    {i+1}. {feature['feature']}: {feature['importance']:.4f}")
        
        # Save results to file if specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to {args.output}")
        
        print(f"\nModels saved with base name: {args.model_name}")
        print("Use predict_ci.py to make predictions with these models")
        print(f"\n💡 Tip: Use collect_data.py --data-type ci to refresh cached CI data when needed")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 