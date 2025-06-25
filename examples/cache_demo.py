#!/usr/bin/env python3
"""
Demo script showing training cache functionality.

Usage:
    python examples/cache_demo.py --repo "owner/repo" --token "your_github_token"
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add parent directory to path to import our module
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_pr_timeline import Config, GitHubDataCollector, TrainingCache

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    parser = argparse.ArgumentParser(description='Demo training cache functionality')
    parser.add_argument('--repo', required=True, help='Repository name (owner/repo)')
    parser.add_argument('--token', help='GitHub token (or set GITHUB_TOKEN env var)')
    parser.add_argument('--max-prs', type=int, default=20,
                       help='Maximum number of PRs to collect')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear cache for the repository before collecting')
    
    args = parser.parse_args()
    
    # Set up configuration
    config = Config()
    config.github_token = args.token or os.getenv('GITHUB_TOKEN')
    
    if not config.github_token:
        print("Error: GitHub token is required. Set GITHUB_TOKEN environment variable or use --token")
        sys.exit(1)
    
    try:
        # Initialize components
        collector = GitHubDataCollector(config)
        cache = collector.training_cache
        
        print(f"Training cache directory: {cache.cache_dir}")
        print("-" * 60)
        
        # Show initial cache stats
        stats = cache.get_cache_stats()
        print(f"Current cache stats:")
        print(f"  Total repositories: {stats['total_repos']}")
        print(f"  Total cached PRs: {stats['total_prs']}")
        for repo, count in stats['repos'].items():
            print(f"    {repo}: {count} PRs")
        print()
        
        # Clear cache if requested
        if args.clear_cache:
            cleared = cache.clear_repo_cache(args.repo)
            print(f"Cleared {cleared} cached PRs for {args.repo}")
            print()
        
        # Show cached PRs for this repo
        cached_numbers = cache.get_cached_pr_numbers(args.repo)
        print(f"Currently cached PRs for {args.repo}: {len(cached_numbers)}")
        if cached_numbers:
            print(f"  PR numbers: {sorted(list(cached_numbers))}")
        print()
        
        # Reset API call counter for this demo
        collector.reset_api_call_count()
        
        # Collect PR data (will use cache when available)
        print(f"Collecting up to {args.max_prs} PRs from {args.repo}...")
        df = collector.collect_pr_data(args.repo, limit=args.max_prs)
        
        print(f"\nCollection complete!")
        print(f"Total PRs collected: {len(df)}")
        print(f"API calls made during collection: {collector.get_api_call_count()}")
        
        # Show updated cache stats
        stats = cache.get_cache_stats()
        print(f"\nUpdated cache stats:")
        print(f"  Total repositories: {stats['total_repos']}")
        print(f"  Total cached PRs: {stats['total_prs']}")
        for repo, count in stats['repos'].items():
            print(f"    {repo}: {count} PRs")
        
        # Show some sample data
        if len(df) > 0:
            print(f"\nSample PR data:")
            for i, row in df.head(3).iterrows():
                print(f"  PR #{row['pr_number']}: {row['title'][:50]}...")
                print(f"    Merge time: {row.get('merge_time_hours', 'N/A')} hours")
                print(f"    Files changed: {row.get('files_changed', 'N/A')}")
                print()
        
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 