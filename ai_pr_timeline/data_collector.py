"""
Data collection module for fetching PR data from GitHub API.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Generator
import pandas as pd
import requests
from github import Github
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
                    # Only process merged PRs with valid data
                    if pr.merged and pr.merged_at and pr.created_at:
                        if self._is_bot_author(pr):
                            # Log when we skip bot PRs
                            bot_name = pr.user.login if pr.user else "unknown"
                            logger.debug(f"Skipped bot PR #{pr.number} by {bot_name}")
                        else:
                            # Process human-authored PRs
                            data = self._extract_pr_features(pr)
                            pr_data.append(data)
                            count += 1
                            
                            # Log details about each PR processed
                            merge_time_hours = data.get('merge_time_hours', 0)
                            merge_time_days = merge_time_hours / 24
                            pr_title = pr.title[:60] + "..." if len(pr.title) > 60 else pr.title
                            
                            # Calculate total time for comparison
                            total_time_hours = (pr.merged_at - pr.created_at).total_seconds() / 3600
                            draft_time_hours = total_time_hours - merge_time_hours
                            
                            draft_info = f" (excl. {draft_time_hours:.1f}h draft)" if draft_time_hours > 0.1 else ""
                            
                            logger.info(f"Processed PR #{pr.number}: '{pr_title}' | "
                                      f"Merge time: {merge_time_hours:.1f}h ({merge_time_days:.1f}d){draft_info} | "
                                      f"Files: {pr.changed_files}, +{pr.additions}/-{pr.deletions}")
                            
                            if count % 50 == 0:
                                logger.info(f"--- Collected {count} PRs so far ---")
                        
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
    
    def _extract_pr_features(self, pr) -> Dict:
        """
        Extract features from a single PR.
        
        Args:
            pr: GitHub PullRequest object
            
        Returns:
            Dictionary with PR features
        """
        # Calculate merge time in hours (only for merged PRs), excluding draft time
        merge_time_hours = None
        if pr.merged_at and pr.created_at:
            total_time_hours = (pr.merged_at - pr.created_at).total_seconds() / 3600
            draft_time_hours = self._calculate_draft_time(pr)
            merge_time_hours = max(0, total_time_hours - draft_time_hours)  # Ensure non-negative
        
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
        
        # Get author info - handle cases where author_association might not exist
        author_association = getattr(pr, 'author_association', 'NONE')
        
        # Check if it's a draft PR
        is_draft = pr.draft if hasattr(pr, 'draft') else False
        
        result = {
            'pr_number': pr.number,
            'title': pr.title,
            'body': pr.body or '',
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
        
        # Only include merge_time_hours for merged PRs
        if merge_time_hours is not None:
            result['merge_time_hours'] = merge_time_hours
            
        return result
    
    def _is_bot_author(self, pr) -> bool:
        """
        Check if a PR was authored by a bot.
        
        Args:
            pr: GitHub PullRequest object
            
        Returns:
            True if authored by a bot, False otherwise
        """
        try:
            author = pr.user
            if not author:
                return False
                
            # Check if user type is Bot
            if hasattr(author, 'type') and author.type == 'Bot':
                return True
                
            # Check for common bot username patterns
            username = author.login.lower()
            bot_patterns = [
                '[bot]',
                'bot-',
                '-bot',
                'dependabot',
                'renovate',
                'github-actions',
                'codecov',
                'greenkeeper',
                'snyk-bot',
                'whitesource'
            ]
            
            return any(pattern in username for pattern in bot_patterns)
            
        except Exception:
            # If we can't determine, assume it's not a bot
            return False
    
    def _calculate_draft_time(self, pr) -> float:
        """
        Calculate the total time a PR spent in draft state.
        
        Args:
            pr: GitHub PullRequest object
            
        Returns:
            Draft time in hours
        """
        try:
            # Get timeline events for the PR
            timeline = list(pr.get_timeline())
            draft_time_hours = 0.0
            current_draft_start = None
            
            # Check timeline events to determine draft history
            ready_for_review_events = []
            converted_to_draft_events = []
            
            for event in timeline:
                if hasattr(event, 'event'):
                    if event.event == 'ready_for_review':
                        ready_for_review_events.append(event.created_at)
                    elif event.event == 'converted_to_draft':
                        converted_to_draft_events.append(event.created_at)
            
            # If there's a 'ready_for_review' event, PR was created as draft
            if ready_for_review_events:
                current_draft_start = pr.created_at
            
            # Process draft/ready cycles chronologically
            all_events = []
            for timestamp in ready_for_review_events:
                all_events.append(('ready_for_review', timestamp))
            for timestamp in converted_to_draft_events:
                all_events.append(('converted_to_draft', timestamp))
            
            # Sort events by timestamp
            all_events.sort(key=lambda x: x[1])
            
            for event_type, event_time in all_events:
                if event_type == 'converted_to_draft':
                    # PR became draft
                    current_draft_start = event_time
                elif event_type == 'ready_for_review':
                    # PR became ready for review
                    if current_draft_start:
                        draft_period = (event_time - current_draft_start).total_seconds() / 3600
                        draft_time_hours += draft_period
                        current_draft_start = None
            
            # If PR is still draft when merged (shouldn't happen but just in case)
            if current_draft_start and pr.merged_at:
                final_draft_period = (pr.merged_at - current_draft_start).total_seconds() / 3600
                draft_time_hours += final_draft_period
            
            return draft_time_hours
            
        except Exception as e:
            logger.debug(f"Could not calculate draft time for PR #{pr.number}: {e}")
            return 0.0
    
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