#!/usr/bin/env python3
"""
Command-line tool for comprehensive PR and CI predictions.
"""

import argparse
import sys
import os
import json
from pathlib import Path

# Add the parent directory to Python path to import ai_pr_timeline
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_pr_timeline.merge_time.predictor import PRTimelinePredictor
from ai_pr_timeline.ci.predictor import CIPredictor
from ai_pr_timeline.config import Config
from ai_pr_timeline.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description='Comprehensive PR and CI predictions')
    parser.add_argument('--repo', required=True,
                       help='Repository name in format owner/repo')
    parser.add_argument('--pr', type=int, required=True,
                       help='Pull request number')
    parser.add_argument('--pr-model', default='pr_model',
                       help='PR model filename (default: pr_model)')
    parser.add_argument('--ci-model', default='ci_model',
                       help='Base name of CI models (default: ci_model)')
    parser.add_argument('--token', help='GitHub token (or set GITHUB_TOKEN env var)')
    parser.add_argument('--output', help='Output file for prediction results (JSON)')
    parser.add_argument('--skip-pr', action='store_true',
                       help='Skip PR merge time prediction')
    parser.add_argument('--skip-ci', action='store_true',
                       help='Skip CI outcome prediction')
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
    
    combined_results = {
        'repository': args.repo,
        'pr_number': args.pr,
        'pr_prediction': None,
        'ci_predictions': None
    }
    
    print(f"Analyzing PR #{args.pr} in {args.repo}")
    print("="*60)
    
    # PR merge time prediction
    if not args.skip_pr:
        try:
            print("\n🔍 LOADING PR PREDICTION MODEL...")
            pr_predictor = PRTimelinePredictor(config)
            pr_predictor.load_model(args.pr_model)
            
            print("📊 PREDICTING PR MERGE TIME...")
            pr_prediction = pr_predictor.predict_pr_timeline(args.repo, args.pr)
            combined_results['pr_prediction'] = pr_prediction
            
            if 'error' not in pr_prediction:
                print(f"\n⏰ PR MERGE TIME PREDICTION:")
                print(f"  Estimated merge time: {pr_prediction['predicted_hours']:.1f} hours")
                print(f"  Estimated merge time: {pr_prediction['predicted_days']:.1f} days")
                print(f"  Confidence level: {pr_prediction['prediction_confidence']}")
            else:
                print(f"❌ PR prediction error: {pr_prediction['error']}")
                
        except FileNotFoundError:
            print(f"⚠️  PR model '{args.pr_model}' not found. Skipping PR prediction.")
            combined_results['pr_prediction'] = {'error': 'Model not found'}
        except Exception as e:
            print(f"❌ Error in PR prediction: {e}")
            combined_results['pr_prediction'] = {'error': str(e)}
    
    # CI prediction
    if not args.skip_ci:
        try:
            print("\n🔍 LOADING CI PREDICTION MODELS...")
            ci_predictor = CIPredictor(config)
            ci_predictor.load_models(args.ci_model)
            
            model_summary = ci_predictor.get_model_summary()
            if model_summary['model_count'] > 0:
                print(f"✅ Loaded {model_summary['model_count']} CI models: {', '.join(model_summary['trained_models'])}")
                
                print("🧪 PREDICTING CI OUTCOMES...")
                ci_predictions = ci_predictor.predict_ci_timeline(args.repo, args.pr)
                combined_results['ci_predictions'] = ci_predictions
                
                if 'error' not in ci_predictions:
                    pred_data = ci_predictions.get('predictions', {})
                    
                    # Duration predictions
                    if 'duration' in pred_data and 'total_duration_hours' in pred_data['duration']:
                        duration = pred_data['duration']
                        print(f"\n⏱️  CI DURATION PREDICTION:")
                        print(f"  Expected CI runtime: {duration['total_duration_hours']:.1f} hours")
                        print(f"  Expected CI runtime: {duration['total_duration_minutes']:.0f} minutes")
                        print(f"  Confidence: {duration['confidence']}")
                    
                    # Attempts predictions
                    if 'attempts' in pred_data and 'expected_attempts' in pred_data['attempts']:
                        attempts = pred_data['attempts']
                        print(f"\n🔄 CI RETRY PREDICTION:")
                        print(f"  Expected attempts: {attempts['expected_attempts']}")
                        print(f"  Confidence: {attempts['confidence']}")
                    
                    # Success predictions
                    if 'success' in pred_data and 'success_probability' in pred_data['success']:
                        success = pred_data['success']
                        print(f"\n✅ CI SUCCESS PREDICTION:")
                        print(f"  Success probability: {success['success_probability']*100:.1f}%")
                        print(f"  Likely outcome: {success['likely_outcome']}")
                        print(f"  Confidence: {success['confidence']}")
                    
                    # Summary
                    if 'summary' in ci_predictions:
                        summary = ci_predictions['summary']
                        print(f"\n📋 CI SUMMARY:")
                        for key, value in summary.items():
                            key_formatted = key.replace('_', ' ').title()
                            print(f"  {key_formatted}: {value}")
                else:
                    print(f"❌ CI prediction error: {ci_predictions['error']}")
            else:
                print("⚠️  No CI models loaded. Skipping CI prediction.")
                combined_results['ci_predictions'] = {'error': 'No models loaded'}
                
        except FileNotFoundError:
            print(f"⚠️  CI models with base name '{args.ci_model}' not found. Skipping CI prediction.")
            combined_results['ci_predictions'] = {'error': 'Models not found'}
        except Exception as e:
            print(f"❌ Error in CI prediction: {e}")
            combined_results['ci_predictions'] = {'error': str(e)}
    
    # Combined analysis
    print("\n" + "="*60)
    print("📈 COMBINED ANALYSIS")
    print("="*60)
    
    pr_pred = combined_results.get('pr_prediction')
    ci_pred = combined_results.get('ci_predictions')
    
    if pr_pred and 'predicted_hours' in pr_pred and ci_pred and 'predictions' in ci_pred:
        pr_hours = pr_pred['predicted_hours']
        ci_data = ci_pred['predictions']
        
        print(f"\n🔄 TIMELINE COMPARISON:")
        if 'duration' in ci_data and 'total_duration_hours' in ci_data['duration']:
            ci_hours = ci_data['duration']['total_duration_hours']
            print(f"  PR merge time: {pr_hours:.1f} hours")
            print(f"  CI completion: {ci_hours:.1f} hours")
            
            if ci_hours > pr_hours:
                print(f"  ⚠️  CI may take longer than PR merge process")
            else:
                print(f"  ✅ CI should complete before PR merge")
        
        print(f"\n💡 RECOMMENDATIONS:")
        if ci_pred and 'summary' in ci_pred:
            ci_summary = ci_pred['summary']
            if 'recommendation' in ci_summary:
                print(f"  CI: {ci_summary['recommendation']}")
        
        if pr_pred['prediction_confidence'] == 'high':
            print(f"  PR: High confidence in merge time prediction")
        elif pr_pred['prediction_confidence'] == 'medium':
            print(f"  PR: Medium confidence - monitor for changes")
        else:
            print(f"  PR: Low confidence - prediction may be inaccurate")
    
    # Save results to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(combined_results, f, indent=2, default=str)
        print(f"\n💾 Results saved to {args.output}")
    
    print(f"\n🎯 Analysis complete for PR #{args.pr} in {args.repo}")


if __name__ == "__main__":
    main() 