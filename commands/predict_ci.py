#!/usr/bin/env python3
"""
Command-line tool for predicting CI outcomes.
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
    parser = argparse.ArgumentParser(description='Predict CI outcomes for PRs')
    parser.add_argument('--repo', required=True,
                       help='Repository name in format owner/repo')
    parser.add_argument('--pr', type=int, required=True,
                       help='Pull request number')
    parser.add_argument('--model-name', default='ci_model',
                       help='Base name of trained models to load (default: ci_model)')
    parser.add_argument('--token', help='GitHub token (or set GITHUB_TOKEN env var)')
    parser.add_argument('--output', help='Output file for prediction results (JSON)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Set up configuration
    config = Config()
    config.github_token = args.token or os.getenv('GITHUB_TOKEN')
    
    if not config.github_token:
        print("Error: GitHub token is required. Set GITHUB_TOKEN environment variable or use --token")
        sys.exit(1)
    
    try:
        # Initialize CI predictor and load models
        ci_predictor = CIPredictor(config)
        
        print(f"Loading CI models with base name: {args.model_name}")
        ci_predictor.load_models(args.model_name)
        
        model_summary = ci_predictor.get_model_summary()
        print(f"Loaded {model_summary['model_count']} models: {', '.join(model_summary['trained_models'])}")
        
        # Make prediction
        print(f"\nPredicting CI outcomes for PR #{args.pr} in {args.repo}")
        predictions = ci_predictor.predict_ci_timeline(args.repo, args.pr)
        
        # Display results
        print("\n" + "="*60)
        print("CI PREDICTION RESULTS")
        print("="*60)
        
        if 'error' in predictions:
            print(f"Error: {predictions['error']}")
            sys.exit(1)
        
        print(f"Repository: {predictions['repository']}")
        print(f"PR Number: #{predictions['pr_number']}")
        
        pred_data = predictions.get('predictions', {})
        
        # Duration predictions
        if 'duration' in pred_data and 'total_duration_hours' in pred_data['duration']:
            duration = pred_data['duration']
            print(f"\n📅 DURATION PREDICTION:")
            print(f"  Expected total time: {duration['total_duration_hours']:.1f} hours")
            print(f"  Expected total time: {duration['total_duration_minutes']:.0f} minutes")
            print(f"  Confidence: {duration['confidence']}")
        
        # Attempts predictions
        if 'attempts' in pred_data and 'expected_attempts' in pred_data['attempts']:
            attempts = pred_data['attempts']
            print(f"\n🔄 RETRY PREDICTION:")
            print(f"  Expected attempts: {attempts['expected_attempts']}")
            print(f"  Confidence: {attempts['confidence']}")
        
        # Success predictions
        if 'success' in pred_data and 'success_probability' in pred_data['success']:
            success = pred_data['success']
            print(f"\n✅ SUCCESS PREDICTION:")
            print(f"  Success probability: {success['success_probability']*100:.1f}%")
            print(f"  Likely outcome: {success['likely_outcome']}")
            print(f"  Confidence: {success['confidence']}")
        
        # Summary
        if 'summary' in predictions:
            summary = predictions['summary']
            print(f"\n📋 SUMMARY:")
            for key, value in summary.items():
                key_formatted = key.replace('_', ' ').title()
                print(f"  {key_formatted}: {value}")
        
        # Save results to file if specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(predictions, f, indent=2, default=str)
            print(f"\nResults saved to {args.output}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find trained models. Train models first using train_ci_model.py")
        print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 