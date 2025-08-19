#!/usr/bin/env python3
"""
Command-line tool for collecting and caching PR and CI data from repositories.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the parent directory to Python path to import ai_pr_timeline
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_pr_timeline import GitHubDataCollector, Config
from ai_pr_timeline.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description='Collect and cache PR and CI data from repositories')
    parser.add_argument('--repos', required=True, nargs='+',
                       help='Repository names in format owner/repo')
    parser.add_argument('--token', help='GitHub token (or set GITHUB_TOKEN env var)')
    parser.add_argument('--max-new-prs', type=int, default=50,
                       help='Maximum number of new PRs to fetch from API per repository (default: 50)')
    parser.add_argument('--data-type', default='all', choices=['all', 'pr', 'ci'],
                       help='Type of data to collect: all (default), pr (PR data only), or ci (CI data only)')
    parser.add_argument('--force-refresh', action='store_true',
                       help='Force refresh of all data (ignore cache)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--cache-dir', type=str,
                       help='Directory to store cached data (default: training_cache)')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Determine what data to collect
    collect_pr_data = args.data_type in ['all', 'pr']
    collect_ci_data = args.data_type in ['all', 'ci']
    
    # Set up configuration
    config = Config()
    config.github_token = args.token or os.getenv('GITHUB_TOKEN')
    
    # Override cache directory if provided
    if args.cache_dir:
        config.training_cache_dir = args.cache_dir
    
    if not config.github_token:
        print("Error: GitHub token is required. Set GITHUB_TOKEN environment variable or use --token")
        sys.exit(1)
    
    try:
        # Initialize data collector
        data_collector = GitHubDataCollector(config)
        
        print(f"🔄 Collecting data from repositories: {', '.join(args.repos)}")
        print(f"🆕 Max new PRs from API per repository: {args.max_new_prs}")
        print(f"📊 Data type: {args.data_type}")
        print(f"💾 Cache directory: {config.training_cache_dir}")
        if args.force_refresh:
            print("🔄 Force refresh enabled - ignoring cache")
        print("-" * 60)
        
        # Reset API call counter
        data_collector.reset_api_call_count()
        
        # Clear cache if force refresh is enabled
        if args.force_refresh:
            for repo_name in args.repos:
                print(f"🗑️  Clearing cache for {repo_name}")
                data_collector.training_cache.clear_repo_cache(repo_name)
        
        # Collect data from repositories
        total_pr_count = 0
        total_ci_count = 0
        
        for i, repo_name in enumerate(args.repos, 1):
            print(f"\n📁 Processing repository {i}/{len(args.repos)}: {repo_name}")
            
            try:
                pr_df, ci_df = data_collector.collect_all_data(
                    repo_name, 
                    limit=None,  # No limit on total PRs, only on new API calls
                    max_new_prs=args.max_new_prs,
                    collect_pr_data=collect_pr_data,
                    collect_ci_data=collect_ci_data
                )
                
                repo_pr_count = len(pr_df) if not pr_df.empty else 0
                repo_ci_count = len(ci_df) if not ci_df.empty else 0
                
                total_pr_count += repo_pr_count
                total_ci_count += repo_ci_count
                
                print(f"   ✅ Collected {repo_pr_count} PRs and {repo_ci_count} CI runs")
                
            except Exception as e:
                print(f"   ❌ Error collecting from {repo_name}: {e}")
                continue
        
        # Final summary
        print("\n" + "="*60)
        print("📊 DATA COLLECTION SUMMARY")
        print("="*60)
        print(f"Repositories processed: {len(args.repos)}")
        print(f"Total PRs collected: {total_pr_count}")
        print(f"Total CI runs collected: {total_ci_count}")
        print(f"Total API calls made: {data_collector.get_api_call_count()}")
        
        # Show cache statistics
        cache_stats = data_collector.training_cache.get_cache_stats()
        print(f"\n💾 Cache Statistics:")
        print(f"Total repositories in cache: {cache_stats.get('total_repositories', 0)}")
        print(f"Total PRs in cache: {cache_stats.get('total_prs', 0)}")
        
        print(f"\n✅ Data collection completed successfully!")
        print(f"💾 Data stored in: {config.training_cache_dir}")
        print(f"💡 Use train_model.py or train_ci_model.py to train models with this cached data")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 