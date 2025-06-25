#!/usr/bin/env python3
"""
Example script for predicting PR timeline using a trained model.

Usage:
    python examples/predict_pr.py --repo "owner/repo" --pr-number 123 --model "model_file.pkl"
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
    parser = argparse.ArgumentParser(description='Predict PR timeline')
    parser.add_argument('--repo', required=True, help='Repository name (owner/repo)')
    parser.add_argument('--pr-number', type=int, required=True, help='PR number to predict')
    parser.add_argument('--model', required=True, help='Trained model filename')
    parser.add_argument('--token', help='GitHub token (or set GITHUB_TOKEN env var)')
    
    args = parser.parse_args()
    
    # Set up configuration
    config = Config()
    config.github_token = args.token or os.getenv('GITHUB_TOKEN')
    
    if not config.github_token:
        print("Error: GitHub token is required. Set GITHUB_TOKEN environment variable or use --token")
        sys.exit(1)
    
    try:
        # Initialize predictor
        predictor = PRTimelinePredictor(config)
        
        # Load trained model
        print(f"Loading model: {args.model}")
        predictor.load_model(args.model)
        
        # Get model info
        model_info = predictor.get_model_metrics()
        print(f"Model type: {model_info['model_type']}")
        print(f"Features used: {model_info['feature_count']}")
        print("-" * 50)
        
        # Make prediction
        print(f"Predicting timeline for PR #{args.pr_number} in {args.repo}")
        
        result = predictor.predict_pr_timeline(args.repo, args.pr_number)
        
        print(f"\nPR: {result['pr_title']}")
        print(f"Repository: {result['repository']}")
        print(f"PR Number: #{result['pr_number']}")
        
        print(f"\nPredicted Timeline:")
        print(f"  Hours: {result['predicted_hours']:.1f}")
        print(f"  Days: {result['predicted_days']:.1f}")
        print(f"  Category: {result['time_category']}")
        
        print(f"\nConfidence Interval (95%):")
        ci = result['confidence_interval_hours']
        print(f"  {ci['lower']:.1f} - {ci['upper']:.1f} hours")
        print(f"  {ci['lower']/24:.1f} - {ci['upper']/24:.1f} days")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 