"""
Data collection module for fetching both PR and CI data from GitHub API.
"""

import time
import logging
from typing import List, Dict, Optional, Tuple
import pandas as pd
import requests
from github import Github
from github.GithubException import RateLimitExceededException
from datetime import datetime, timedelta

from ..config import Config, DEFAULT_CONFIG
from .training_cache import TrainingCache

logger = logging.getLogger(__name__)


class GitHubDataCollector:
    """Enhanced collector for both PR and CI data from GitHub repositories."""

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
        
        # Initialize training cache
        self.training_cache = TrainingCache(config.training_cache_dir)
        
        # API call tracking
        self.api_call_count = 0

    def _log_api_call(self, endpoint: str, description: str = "") -> None:
        """Log an API call with counter."""
        self.api_call_count += 1
        desc_suffix = f" - {description}" if description else ""
        logger.info(f"API Call #{self.api_call_count}: {endpoint}{desc_suffix}")

    def get_api_call_count(self) -> int:
        """Get the current API call count."""
        return self.api_call_count

    def reset_api_call_count(self) -> None:
        """Reset the API call counter."""
        self.api_call_count = 0

    def collect_all_data(self, repo_name: str, limit: Optional[int] = None, 
                        max_new_prs: Optional[int] = None,
                        collect_pr_data: bool = True,
                        collect_ci_data: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Collect both PR and CI data from a repository in a single pass.

        Args:
            repo_name: Repository name in format 'owner/repo'
            limit: Maximum number of PRs to collect
            max_new_prs: Maximum number of new PRs to fetch from API
            collect_pr_data: Whether to collect PR merge time data
            collect_ci_data: Whether to collect CI data

        Returns:
            Tuple of (pr_dataframe, ci_dataframe)
        """
        logger.info(f"Collecting data from {repo_name}")
        logger.info(f"Collecting PR data: {collect_pr_data}, Collecting CI data: {collect_ci_data}")
        
        # Check cache first
        cached_prs = self.training_cache.get_cached_prs_for_repo(repo_name) if collect_pr_data else []
        cached_pr_numbers = self.training_cache.get_cached_pr_numbers(repo_name) if collect_pr_data else set()
        
        # Load cached CI data
        cached_ci_data = []
        if collect_ci_data:
            try:
                cache_file = f"ci_data_{repo_name.replace('/', '_')}.csv"
                cached_ci_df = self.load_data(cache_file)
                cached_ci_data = [row.to_dict() for _, row in cached_ci_df.iterrows()] if not cached_ci_df.empty else []
                if cached_ci_data:
                    logger.info(f"Found {len(cached_ci_data)} cached CI runs for {repo_name}")
            except Exception:
                logger.debug(f"No cached CI data found for {repo_name}")
        
        if cached_prs:
            logger.info(f"Found {len(cached_prs)} cached PRs for {repo_name}")
        
        pr_data = cached_prs.copy() if collect_pr_data else []
        ci_data = cached_ci_data.copy() if collect_ci_data else []
        
        count = len(pr_data)
        max_prs = limit or self.config.max_prs_per_repo
        max_new = max_new_prs if max_new_prs is not None else max_prs
        
        # Validate max_new_prs parameter
        if max_new > max_prs:
            raise ValueError(f"max_new_prs ({max_new}) cannot be greater than limit ({max_prs})")
        
        # If we have enough cached PR data and only collecting PR data, return early
        if collect_pr_data and not collect_ci_data and count >= max_prs:
            logger.info(f"Using {max_prs} cached PRs from {repo_name} (no API calls needed)")
            pr_df = pd.DataFrame(pr_data[:max_prs])
            return pr_df, pd.DataFrame()

        try:
            self._log_api_call(f"GET /repos/{repo_name}", f"Getting repository info")
            repo = self.github.get_repo(repo_name)
            
            self._log_api_call(f"GET /repos/{repo_name}/pulls", f"Getting PRs for data collection")
            prs = repo.get_pulls(state='all' if collect_ci_data else 'closed', 
                               sort='updated', direction='desc')

            new_prs_processed = 0
            for pr in prs:
                # Stop when we've hit either limit
                if (collect_pr_data and count >= max_prs) or new_prs_processed >= max_new:
                    break

                try:
                    # Collect PR data if requested
                    pr_entry = None
                    if collect_pr_data:
                        # Skip if already cached
                        if pr.number in cached_pr_numbers:
                            continue
                        
                        # Only process merged PRs with valid data for PR model
                        if pr.merged and pr.merged_at and pr.created_at:
                            if self._is_bot_author(pr):
                                bot_name = pr.user.login if pr.user else "unknown"
                                logger.debug(f"Skipped bot PR #{pr.number} by {bot_name}")
                                continue

                            pr_entry = self._extract_pr_features(pr)
                            pr_data.append(pr_entry)
                            count += 1

                            # Cache the new PR data
                            self.training_cache.cache_pr(repo_name, pr_entry)

                    # Collect CI data if requested
                    if collect_ci_data:
                        pr_ci_data = self._extract_pr_ci_data(pr, repo_name)
                        if pr_ci_data:
                            ci_data.extend(pr_ci_data)

                    new_prs_processed += 1
                    
                    # Rate limiting
                    if new_prs_processed % 10 == 0:
                        time.sleep(1)

                except Exception as e:
                    logger.warning(f"Error processing PR #{pr.number}: {e}")
                    continue

        except RateLimitExceededException:
            logger.warning("GitHub API rate limit exceeded. Waiting...")
            time.sleep(60)
            return self.collect_all_data(repo_name, limit, max_new_prs, collect_pr_data, collect_ci_data)

        # Cache CI data if collected
        if collect_ci_data and len(ci_data) > 0:
            try:
                cache_file = f"ci_data_{repo_name.replace('/', '_')}.csv"
                cache_df = pd.DataFrame(ci_data)
                self.save_data(cache_df, cache_file)
            except Exception as e:
                logger.warning(f"Could not cache CI data: {e}")

        # Create DataFrames
        pr_df = pd.DataFrame(pr_data) if collect_pr_data else pd.DataFrame()
        ci_df = pd.DataFrame(ci_data) if collect_ci_data else pd.DataFrame()
        
        logger.info(f"Collected {len(pr_df)} PRs and {len(ci_df)} CI runs from {repo_name}")
        logger.info(f"Total API calls made in this session: {self.api_call_count}")
        
        return pr_df, ci_df

    def collect_multiple_repos(self, repo_names: List[str],
                             limit_per_repo: Optional[int] = None,
                             max_new_prs_per_repo: Optional[int] = None,
                             collect_pr_data: bool = True,
                             collect_ci_data: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Collect data from multiple repositories.

        Args:
            repo_names: List of repository names
            limit_per_repo: Maximum number of PRs to collect per repository
            max_new_prs_per_repo: Maximum number of new PRs to fetch from API per repository
            collect_pr_data: Whether to collect PR data
            collect_ci_data: Whether to collect CI data

        Returns:
            Tuple of (combined_pr_dataframe, combined_ci_dataframe)
        """
        all_pr_data = []
        all_ci_data = []

        for repo_name in repo_names:
            try:
                pr_df, ci_df = self.collect_all_data(
                    repo_name, limit_per_repo, max_new_prs_per_repo,
                    collect_pr_data, collect_ci_data
                )
                if not pr_df.empty and collect_pr_data:
                    all_pr_data.append(pr_df)
                if not ci_df.empty and collect_ci_data:
                    all_ci_data.append(ci_df)
            except Exception as e:
                logger.error(f"Failed to collect data from {repo_name}: {e}")
                continue

        # Combine data
        combined_pr_df = pd.concat(all_pr_data, ignore_index=True) if all_pr_data else pd.DataFrame()
        combined_ci_df = pd.concat(all_ci_data, ignore_index=True) if all_ci_data else pd.DataFrame()
        
        logger.info(f"Collected total of {len(combined_pr_df)} PRs and {len(combined_ci_df)} CI runs from {len(repo_names)} repositories")
        
        return combined_pr_df, combined_ci_df

    def collect_pr_data_only(self, repo_name: str, limit: Optional[int] = None, 
                           max_new_prs: Optional[int] = None) -> pd.DataFrame:
        """Collect only PR data (for backward compatibility)."""
        pr_df, _ = self.collect_all_data(repo_name, limit, max_new_prs, 
                                        collect_pr_data=True, collect_ci_data=False)
        return pr_df

    def collect_ci_data_only(self, repo_name: str, limit: Optional[int] = None, 
                           max_new_prs: Optional[int] = None) -> pd.DataFrame:
        """Collect only CI data (for backward compatibility)."""
        _, ci_df = self.collect_all_data(repo_name, limit, max_new_prs, 
                                        collect_pr_data=False, collect_ci_data=True)
        return ci_df

    def _extract_pr_features(self, pr) -> Dict:
        """
        Extract features from a single PR (same as original data_collector).
        """
        # Calculate merge time in hours (only for merged PRs), excluding draft time
        merge_time_hours = None
        if pr.merged_at and pr.created_at:
            total_time_hours = (pr.merged_at - pr.created_at).total_seconds() / 3600
            draft_time_hours = self._calculate_draft_time(pr)
            merge_time_hours = max(0, total_time_hours - draft_time_hours)

        # Get review data
        self._log_api_call(f"GET /repos/{pr.base.repo.full_name}/pulls/{pr.number}/reviews", f"PR #{pr.number} reviews")
        reviews = list(pr.get_reviews())
        review_count = len(reviews)

        # Get comment data
        self._log_api_call(f"GET /repos/{pr.base.repo.full_name}/issues/{pr.number}/comments", f"PR #{pr.number} comments")
        comments = list(pr.get_issue_comments())
        comment_count = len(comments)

        # Get commit data
        self._log_api_call(f"GET /repos/{pr.base.repo.full_name}/pulls/{pr.number}/commits", f"PR #{pr.number} commits")
        commits = list(pr.get_commits())
        commit_count = len(commits)

        # Calculate time features
        created_hour = pr.created_at.hour
        created_day = pr.created_at.weekday()

        # Get file changes
        files_changed = pr.changed_files
        additions = pr.additions
        deletions = pr.deletions

        # Get author info
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

    def _extract_pr_ci_data(self, pr, repo_name: str) -> List[Dict]:
        """
        Extract CI data from a single PR (same as ci_data_collector).
        """
        ci_runs = []
        
        try:
            # Get status checks
            commits = list(pr.get_commits())
            if not commits:
                return ci_runs
                
            # Focus on the latest commit for CI data
            latest_commit = commits[-1]
            
            # Get status checks
            self._log_api_call(f"GET /repos/{repo_name}/statuses/{latest_commit.sha}", 
                             f"PR #{pr.number} status checks")
            statuses = list(latest_commit.get_statuses())
            
            # Get check runs (newer GitHub Actions format)
            try:
                self._log_api_call(f"GET /repos/{repo_name}/commits/{latest_commit.sha}/check-runs", 
                                 f"PR #{pr.number} check runs")
                check_runs = list(latest_commit.get_check_runs())
            except Exception as e:
                logger.debug(f"Could not get check runs for PR #{pr.number}: {e}")
                check_runs = []

            # Process status checks
            for status in statuses:
                ci_run_data = self._extract_status_data(status, pr, repo_name)
                if ci_run_data:
                    ci_runs.append(ci_run_data)

            # Process check runs
            for check_run in check_runs:
                ci_run_data = self._extract_check_run_data(check_run, pr, repo_name)
                if ci_run_data:
                    ci_runs.append(ci_run_data)

        except Exception as e:
            logger.warning(f"Error extracting CI data from PR #{pr.number}: {e}")

        return ci_runs

    def _extract_status_data(self, status, pr, repo_name: str) -> Optional[Dict]:
        """Extract data from a GitHub status check."""
        try:
            # Calculate duration if available
            duration_seconds = None
            if hasattr(status, 'created_at') and hasattr(status, 'updated_at'):
                if status.created_at and status.updated_at:
                    duration_seconds = (status.updated_at - status.created_at).total_seconds()

            return {
                'repository': repo_name,
                'pr_number': pr.number,
                'pr_title': pr.title,
                'pr_created_at': pr.created_at,
                'pr_merged_at': pr.merged_at,
                'pr_state': pr.state,
                'ci_type': 'status',
                'ci_name': status.context,
                'ci_state': status.state,
                'ci_description': status.description,
                'ci_target_url': status.target_url,
                'ci_created_at': status.created_at,
                'ci_updated_at': status.updated_at,
                'ci_duration_seconds': duration_seconds,
                'pr_files_changed': pr.changed_files,
                'pr_additions': pr.additions,
                'pr_deletions': pr.deletions,
                'pr_commits': pr.commits,
                'pr_author': pr.user.login if pr.user else None,
                'pr_is_draft': getattr(pr, 'draft', False)
            }
        except Exception as e:
            logger.debug(f"Error extracting status data: {e}")
            return None

    def _extract_check_run_data(self, check_run, pr, repo_name: str) -> Optional[Dict]:
        """Extract data from a GitHub check run."""
        try:
            # Calculate duration
            duration_seconds = None
            if check_run.started_at and check_run.completed_at:
                duration_seconds = (check_run.completed_at - check_run.started_at).total_seconds()

            return {
                'repository': repo_name,
                'pr_number': pr.number,
                'pr_title': pr.title,
                'pr_created_at': pr.created_at,
                'pr_merged_at': pr.merged_at,
                'pr_state': pr.state,
                'ci_type': 'check_run',
                'ci_name': check_run.name,
                'ci_state': check_run.conclusion or check_run.status,
                'ci_description': check_run.output.summary if check_run.output else None,
                'ci_target_url': check_run.html_url,
                'ci_created_at': check_run.started_at,
                'ci_updated_at': check_run.completed_at,
                'ci_duration_seconds': duration_seconds,
                'pr_files_changed': pr.changed_files,
                'pr_additions': pr.additions,
                'pr_deletions': pr.deletions,
                'pr_commits': pr.commits,
                'pr_author': pr.user.login if pr.user else None,
                'pr_is_draft': getattr(pr, 'draft', False)
            }
        except Exception as e:
            logger.debug(f"Error extracting check run data: {e}")
            return None

    def _is_bot_author(self, pr) -> bool:
        """Check if a PR was authored by a bot."""
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
                '[bot]', 'bot-', '-bot', 'dependabot', 'renovate',
                'github-actions', 'codecov', 'greenkeeper', 'snyk',
                'renovate-bot', 'dependabot-preview', 'allcontributors',
                'imgbot', 'semantic-release-bot'
            ]

            return any(pattern in username for pattern in bot_patterns)

        except Exception as e:
            logger.warning(f"Error checking if PR author is bot: {e}")
            return False

    def _calculate_draft_time(self, pr) -> float:
        """Calculate total time the PR spent in draft state."""
        try:
            # Get timeline events
            self._log_api_call(f"GET /repos/{pr.base.repo.full_name}/issues/{pr.number}/events", 
                             f"PR #{pr.number} timeline events")
            events = list(pr.get_issue_events())
            timeline_events = []

            for event in events:
                if event.event in ['ready_for_review', 'converted_to_draft']:
                    timeline_events.append({
                        'event': event.event,
                        'created_at': event.created_at
                    })

            # Sort events by time
            timeline_events.sort(key=lambda x: x['created_at'])

            # Calculate draft periods
            draft_time_hours = 0.0
            is_draft = pr.draft if hasattr(pr, 'draft') else False

            # Check if PR was originally created as draft
            if is_draft or any(event['event'] == 'ready_for_review' for event in timeline_events):
                draft_start = pr.created_at
                current_state_is_draft = True

                for event in timeline_events:
                    if event['event'] == 'ready_for_review' and current_state_is_draft:
                        # End of draft period
                        draft_duration = (event['created_at'] - draft_start).total_seconds() / 3600
                        draft_time_hours += draft_duration
                        current_state_is_draft = False
                    elif event['event'] == 'converted_to_draft' and not current_state_is_draft:
                        # Start of new draft period
                        draft_start = event['created_at']
                        current_state_is_draft = True

                # If still in draft at merge time, add the final period
                if current_state_is_draft and pr.merged_at:
                    final_duration = (pr.merged_at - draft_start).total_seconds() / 3600
                    draft_time_hours += final_duration

            return draft_time_hours

        except Exception as e:
            logger.warning(f"Error calculating draft time for PR #{pr.number}: {e}")
            return 0.0

    def get_ci_summary(self, df: pd.DataFrame) -> Dict:
        """Generate a summary of CI data."""
        if df.empty:
            return {}

        summary = {
            'total_ci_runs': len(df),
            'unique_repositories': df['repository'].nunique(),
            'unique_prs': df['pr_number'].nunique(),
            'unique_ci_tests': df['ci_name'].nunique(),
        }

        # Success rates by state
        if 'ci_state' in df.columns:
            state_counts = df['ci_state'].value_counts()
            total_runs = len(df)
            if total_runs > 0:
                success_count = state_counts.get('success', 0) or 0
                failure_count = state_counts.get('failure', 0) or 0
                error_count = state_counts.get('error', 0) or 0
                summary['success_rate'] = success_count / total_runs
                summary['failure_rate'] = (failure_count + error_count) / total_runs
            else:
                summary['success_rate'] = 0.0
                summary['failure_rate'] = 0.0
            summary['state_distribution'] = state_counts.to_dict()

        # Duration statistics
        if 'ci_duration_seconds' in df.columns:
            duration_data = df['ci_duration_seconds'].dropna()
            if not duration_data.empty:
                summary['avg_duration_seconds'] = duration_data.mean()
                summary['median_duration_seconds'] = duration_data.median()
                summary['max_duration_seconds'] = duration_data.max()
                summary['min_duration_seconds'] = duration_data.min()

        # Most common CI tests
        if 'ci_name' in df.columns:
            summary['top_ci_tests'] = df['ci_name'].value_counts().head(10).to_dict()

        return summary

    def save_data(self, df: pd.DataFrame, filename: str) -> None:
        """Save collected data to file."""
        filepath = f"{self.config.data_dir}/{filename}"
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")

    def load_data(self, filename: str) -> pd.DataFrame:
        """Load data from file."""
        filepath = f"{self.config.data_dir}/{filename}"
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded from {filepath}")
        return df 