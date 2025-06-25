#!/usr/bin/env python3
"""
Example script for batch prediction of PR timelines.

Usage:
    python examples/batch_predict.py --repos-file repos.txt --model model.pkl --output results.csv
"""

import os
import sys
import csv
import logging
import argparse
from pathlib import Path
from typing import List

# Add parent directory to path to import our module
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_pr_timeline import PRTimelinePredictor, Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def load_repositories(file_path: str) -> List[str]:
    """Load repository names from file."""
    repos = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                repos.append(line)
    return repos

def main():
    parser = argparse.ArgumentParser(description='Batch predict PR timelines')
    parser.add_argument('--repos-file', required=True, 
                       help='File containing repository names (one per line)')
    parser.add_argument('--model', required=True, help='Trained model filename')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--token', help='GitHub token (or set GITHUB_TOKEN env var)')
    parser.add_argument('--limit-per-repo', type=int, default=50,
                       help='Maximum number of open PRs to process per repository')
    
    args = parser.parse_args()
    
    # Set up configuration
    config = Config()
    config.github_token = args.token or os.getenv('GITHUB_TOKEN')
    
    if not config.github_token:
        print("Error: GitHub token is required. Set GITHUB_TOKEN environment variable or use --token")
        sys.exit(1)
    
    try:
        # Load repositories
        repos = load_repositories(args.repos_file)
        print(f"Loaded {len(repos)} repositories")
        
        # Initialize predictor
        predictor = PRTimelinePredictor(config)
        
        # Load trained model
        print(f"Loading model: {args.model}")
        predictor.load_model(args.model)
        
        # Prepare output file
        with open(args.output, 'w', newline='') as csvfile:
            fieldnames = [
                'repository', 'pr_number', 'pr_title', 'pr_url',
                'predicted_hours', 'predicted_days', 'time_category',
                'confidence_lower', 'confidence_upper', 'created_at'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            total_predictions = 0
            
            for repo_name in repos:
                print(f"\nProcessing repository: {repo_name}")
                
                try:
                    # Get open PRs from the repository
                    repo = predictor.data_collector.github.get_repo(repo_name)
                    open_prs = repo.get_pulls(state='open', sort='created', direction='desc')
                    
                    pr_count = 0
                    for pr in open_prs:
                        if pr_count >= args.limit_per_repo:
                            break
                        
                        try:
                            # Extract PR features (excluding merge time)
                            pr_data = predictor.data_collector._extract_pr_features(pr)
                            if 'merge_time_hours' in pr_data:
                                del pr_data['merge_time_hours']
                            
                            # Make prediction
                            result = predictor.predict_pr_timeline(repo_name, pr.number)
                            
                            # Prepare row data
                            row = {
                                'repository': repo_name,
                                'pr_number': pr.number,
                                'pr_title': pr.title,
                                'pr_url': pr.html_url,
                                'predicted_hours': result['predicted_hours'],
                                'predicted_days': result['predicted_days'],
                                'time_category': result['time_category'],
                                'confidence_lower': result['confidence_interval_hours']['lower'],
                                'confidence_upper': result['confidence_interval_hours']['upper'],
                                'created_at': pr.created_at.isoformat()
                            }
                            
                            writer.writerow(row)
                            pr_count += 1
                            total_predictions += 1
                            
                            if pr_count % 10 == 0:
                                print(f"  Processed {pr_count} PRs...")
                                
                        except Exception as e:
                            print(f"  Error processing PR #{pr.number}: {e}")
                            continue
                    
                    print(f"  Completed {pr_count} predictions for {repo_name}")
                    
                except Exception as e:
                    print(f"  Error processing repository {repo_name}: {e}")
                    continue
        
        print(f"\nBatch prediction completed!")
        print(f"Total predictions: {total_predictions}")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"Error during batch prediction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 