"""
Data collection module for fetching PR data from GitHub API.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Generator
import pandas as pd
import requests
from github import Github, PullRequest
from github.GithubException import RateLimitExceededException

from .config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

class GitHubDataCollector:
    """Collects PR data from GitHub repositories."""
    
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        if not config.github_token:
            raise ValueError("GitHub token is required. Set GITHUB_TOKEN environment variable.")
        
        self.github = Github(config.github_token)
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'token {config.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        })
    
    def collect_pr_data(self, repo_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Collect PR data from a specific repository.
        
        Args:
            repo_name: Repository name in format 'owner/repo'
            limit: Maximum number of PRs to collect
            
        Returns:
            DataFrame with PR data
        """
        logger.info(f"Collecting PR data from {repo_name}")
        
        try:
            repo = self.github.get_repo(repo_name)
            prs = repo.get_pulls(state='closed', sort='updated', direction='desc')
            
            pr_data = []
            count = 0
            max_prs = limit or self.config.max_prs_per_repo
            
            for pr in prs:
                if count >= max_prs:
                    break
                
                try:
                    # Only collect merged PRs with valid data
                    if pr.merged and pr.merged_at and pr.created_at:
                        data = self._extract_pr_features(pr)
                        pr_data.append(data)
                        count += 1
                        
                        if count % 50 == 0:
                            logger.info(f"Collected {count} PRs...")
                            
                except Exception as e:
                    logger.warning(f"Error processing PR #{pr.number}: {e}")
                    continue
                    
        except RateLimitExceededException:
            logger.warning("GitHub API rate limit exceeded. Waiting...")
            time.sleep(60)
            return self.collect_pr_data(repo_name, limit)
        
        df = pd.DataFrame(pr_data)
        logger.info(f"Collected {len(df)} PRs from {repo_name}")
        return df
    
    def _extract_pr_features(self, pr: PullRequest) -> Dict:
        """
        Extract features from a single PR.
        
        Args:
            pr: GitHub PullRequest object
            
        Returns:
            Dictionary with PR features
        """
        # Calculate merge time in hours
        merge_time_hours = (pr.merged_at - pr.created_at).total_seconds() / 3600
        
        # Get review data
        reviews = list(pr.get_reviews())
        review_count = len(reviews)
        
        # Get comment data
        comments = list(pr.get_issue_comments())
        comment_count = len(comments)
        
        # Get commit data
        commits = list(pr.get_commits())
        commit_count = len(commits)
        
        # Calculate time features
        created_hour = pr.created_at.hour
        created_day = pr.created_at.weekday()  # 0=Monday, 6=Sunday
        
        # Get file changes
        files_changed = pr.changed_files
        additions = pr.additions
        deletions = pr.deletions
        
        # Get author info
        author_association = pr.author_association
        
        # Check if it's a draft PR
        is_draft = pr.draft if hasattr(pr, 'draft') else False
        
        return {
            'pr_number': pr.number,
            'title': pr.title,
            'body': pr.body or '',
            'merge_time_hours': merge_time_hours,
            'review_count': review_count,
            'comment_count': comment_count,
            'commit_count': commit_count,
            'files_changed': files_changed,
            'additions': additions,
            'deletions': deletions,
            'created_hour': created_hour,
            'created_day': created_day,
            'author_association': author_association,
            'is_draft': is_draft,
            'created_at': pr.created_at,
            'merged_at': pr.merged_at,
            'repository': pr.base.repo.full_name
        }
    
    def collect_multiple_repos(self, repo_names: List[str], limit_per_repo: Optional[int] = None) -> pd.DataFrame:
        """
        Collect PR data from multiple repositories.
        
        Args:
            repo_names: List of repository names in format 'owner/repo'
            limit_per_repo: Maximum number of PRs to collect per repository
            
        Returns:
            Combined DataFrame with PR data from all repositories
        """
        all_data = []
        
        for repo_name in repo_names:
            try:
                repo_data = self.collect_pr_data(repo_name, limit_per_repo)
                all_data.append(repo_data)
                
                # Rate limiting between repositories
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error collecting data from {repo_name}: {e}")
                continue
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total PRs collected: {len(combined_df)}")
            return combined_df
        else:
            return pd.DataFrame()
    
    def save_data(self, df: pd.DataFrame, filename: str) -> None:
        """Save DataFrame to CSV file."""
        filepath = f"{self.config.data_dir}/{filename}"
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load DataFrame from CSV file."""
        filepath = f"{self.config.data_dir}/{filename}"
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded from {filepath}")
        return df 