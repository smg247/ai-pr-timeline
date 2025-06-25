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
        
        # Display model information
        print("Model loaded successfully!")
        if hasattr(predictor.model_trainer, 'feature_names') and predictor.model_trainer.feature_names:
            print(f"Features used: {len(predictor.model_trainer.feature_names)}")
        print("-" * 50)
        
        # Make prediction
        print(f"Predicting timeline for PR #{args.pr_number} in {args.repo}")
        
        result = predictor.predict_pr_timeline(args.repo, args.pr_number)
        
        print(f"\nPR #{result['pr_number']} in {result['repository']}")
        
        print(f"\nPredicted Timeline:")
        print(f"  Hours: {result['predicted_hours']:.1f}")
        print(f"  Days: {result['predicted_days']:.1f}")
        print(f"  Confidence: {result['prediction_confidence']}")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 