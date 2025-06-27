"""
Data collection module for fetching both PR and CI data from GitHub API.
"""

import time
import logging
from typing import List, Dict, Optional, Tuple, Any
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

    def __init__(self, config: Config = DEFAULT_CONFIG, cache_only: bool = False):
        self.config = config
        self.cache_only = cache_only
        
        if not cache_only:
            if not config.github_token:
                raise ValueError("GitHub token is required for API access. Set GITHUB_TOKEN environment variable or use cache_only=True for training.")

            self.github = Github(config.github_token)
            self.session = requests.Session()
            self.session.headers.update({
                'Authorization': f'token {config.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            })
        else:
            # Cache-only mode - no API clients needed
            self.github = None
            self.session = None
        
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
        
        # Load cached CI data from training cache
        cached_ci_data = []
        cached_ci_pr_numbers = set()
        if collect_ci_data:
            cached_ci_data = self.training_cache.get_all_cached_ci_data(repo_name)
            cached_ci_pr_numbers = self.training_cache.get_cached_pr_numbers_with_ci(repo_name)
            if cached_ci_data:
                logger.info(f"🔧 Found {len(cached_ci_data)} cached CI runs for {repo_name} (from {len(cached_ci_pr_numbers)} PRs)")
        
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
        
        # If in cache-only mode, can't make API calls
        if self.cache_only:
            logger.info(f"Cache-only mode: Using {len(pr_data)} cached PRs and {len(ci_data)} cached CI runs from {repo_name}")
            pr_df = pd.DataFrame(pr_data) if collect_pr_data else pd.DataFrame()
            ci_df = pd.DataFrame(ci_data) if collect_ci_data else pd.DataFrame()
            return pr_df, ci_df

        try:
            self._log_api_call(f"GET /repos/{repo_name}", f"Getting repository info")
            repo = self.github.get_repo(repo_name)  # type: ignore
            
            self._log_api_call(f"GET /repos/{repo_name}/pulls", f"Getting PRs for data collection")
            prs = repo.get_pulls(state='all' if collect_ci_data else 'closed', 
                               sort='updated', direction='desc')

            new_prs_processed = 0
            for pr in prs:
                # Stop when we've hit either limit
                if (collect_pr_data and count >= max_prs) or new_prs_processed >= max_new:
                    break

                try:
                    # Skip bot PRs entirely (for both PR and CI data)
                    if self._is_bot_author(pr):
                        bot_name = pr.user.login if pr.user else "unknown"
                        logger.debug(f"Skipped bot PR #{pr.number} by {bot_name}")
                        continue
                    
                    # Collect PR data and CI data together
                    pr_entry = None
                    pr_ci_data = []
                    
                    # Check what data we need to collect
                    need_pr_data = collect_pr_data and pr.number not in cached_pr_numbers
                    need_ci_data = collect_ci_data and pr.number not in cached_ci_pr_numbers
                    
                    # Skip if we don't need either
                    if not need_pr_data and not need_ci_data:
                        continue
                    
                    # For PR data, only process merged PRs
                    if need_pr_data:
                        if pr.merged and pr.merged_at and pr.created_at:
                            pr_entry = self._extract_pr_features(pr)
                            pr_data.append(pr_entry)
                            count += 1
                        else:
                            # Not a merged PR, skip PR data but might still collect CI data
                            need_pr_data = False

                    # Collect CI data if requested
                    if need_ci_data:
                        pr_ci_data = self._extract_pr_ci_data(pr, repo_name)
                        if pr_ci_data:
                            ci_data.extend(pr_ci_data)

                    # Cache the data together
                    if need_pr_data and pr_entry:
                        # Cache merged PR with CI data if both were collected
                        self.training_cache.cache_pr(repo_name, pr_entry, pr_ci_data if pr_ci_data else None)
                    elif need_ci_data and pr_ci_data:
                        # If we only collected CI data, try to update existing PR cache or create minimal PR entry
                        existing_pr = self.training_cache.get_cached_pr(repo_name, pr.number)
                        if existing_pr:
                            # Update existing cached PR with CI data
                            self.training_cache.cache_pr(repo_name, existing_pr, pr_ci_data)
                        else:
                            # Create minimal PR entry for CI data storage (for non-merged PRs)
                            minimal_pr_entry = {
                                'pr_number': pr.number,
                                'title': pr.title,
                                'repository': repo_name,
                                'created_at': pr.created_at,
                                'merged_at': pr.merged_at,  # Will be None for non-merged PRs
                                'pr_state': pr.state
                            }
                            self.training_cache.cache_pr(repo_name, minimal_pr_entry, pr_ci_data)

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
        """Extract features from a single PR."""
        merge_time_hours = None
        if pr.merged_at and pr.created_at:
            total_time_hours = (pr.merged_at - pr.created_at).total_seconds() / 3600
            draft_time_hours = self._calculate_draft_time(pr)
            merge_time_hours = max(0, total_time_hours - draft_time_hours)  # Exclude draft time from merge time

        self._log_api_call(f"GET /repos/{pr.base.repo.full_name}/pulls/{pr.number}/reviews", f"PR #{pr.number} reviews")
        reviews = list(pr.get_reviews())
        review_count = len(reviews)

        self._log_api_call(f"GET /repos/{pr.base.repo.full_name}/issues/{pr.number}/comments", f"PR #{pr.number} comments")
        comments = list(pr.get_issue_comments())
        comment_count = len(comments)

        self._log_api_call(f"GET /repos/{pr.base.repo.full_name}/pulls/{pr.number}/commits", f"PR #{pr.number} commits")
        commits = list(pr.get_commits())
        commit_count = len(commits)

        created_hour = pr.created_at.hour
        created_day = pr.created_at.weekday()

        files_changed = pr.changed_files
        additions = pr.additions
        deletions = pr.deletions

        author_association = getattr(pr, 'author_association', 'NONE')

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

        if merge_time_hours is not None:
            result['merge_time_hours'] = merge_time_hours

        return result

    def _extract_pr_ci_data(self, pr, repo_name: str) -> List[Dict]:
        """Extract CI data from a single PR."""
        ci_runs = []
        
        try:
            commits = list(pr.get_commits())
            if not commits:
                return ci_runs
                
            latest_commit = commits[-1]  # Focus on the latest commit for CI data
            
            self._log_api_call(f"GET /repos/{repo_name}/statuses/{latest_commit.sha}", 
                             f"PR #{pr.number} status checks")
            statuses = list(latest_commit.get_statuses())
            
            try:
                self._log_api_call(f"GET /repos/{repo_name}/commits/{latest_commit.sha}/check-runs", 
                                 f"PR #{pr.number} check runs")
                check_runs = list(latest_commit.get_check_runs())
            except Exception as e:
                logger.debug(f"Could not get check runs for PR #{pr.number}: {e}")
                check_runs = []

            for status in statuses:
                ci_run_data = self._extract_status_data(status, pr, repo_name)
                if ci_run_data:
                    ci_runs.append(ci_run_data)

            for check_run in check_runs:
                ci_run_data = self._extract_check_run_data(check_run, pr, repo_name)
                if ci_run_data:
                    ci_runs.append(ci_run_data)

        except Exception as e:
            logger.warning(f"Error extracting CI data from PR #{pr.number}: {e}")

        return ci_runs

    def _extract_status_data(self, status, pr, repo_name: str) -> Optional[Dict]:
        """Extract normalized data from a GitHub status check."""
        try:
            duration_seconds = None
            if hasattr(status, 'created_at') and hasattr(status, 'updated_at'):
                if status.created_at and status.updated_at:
                    duration_seconds = (status.updated_at - status.created_at).total_seconds()

            return {
                'ci_name': status.context,
                'ci_target_url': status.target_url,
                'ci_state': status.state,
                'ci_duration_seconds': duration_seconds
            }
        except Exception as e:
            logger.debug(f"Error extracting status data: {e}")
            return None

    def _extract_check_run_data(self, check_run, pr, repo_name: str) -> Optional[Dict]:
        """Extract normalized data from a GitHub check run."""
        try:
            duration_seconds = None
            if check_run.started_at and check_run.completed_at:
                duration_seconds = (check_run.completed_at - check_run.started_at).total_seconds()

            return {
                'ci_name': check_run.name,
                'ci_target_url': check_run.html_url,
                'ci_state': check_run.conclusion or check_run.status,
                'ci_duration_seconds': duration_seconds
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

            if hasattr(author, 'type') and author.type == 'Bot':
                return True

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

            timeline_events.sort(key=lambda x: x['created_at'])

            draft_time_hours = 0.0
            is_draft = pr.draft if hasattr(pr, 'draft') else False

            if is_draft or any(event['event'] == 'ready_for_review' for event in timeline_events):
                draft_start = pr.created_at
                current_state_is_draft = True

                for event in timeline_events:
                    if event['event'] == 'ready_for_review' and current_state_is_draft:
                        draft_duration = (event['created_at'] - draft_start).total_seconds() / 3600
                        draft_time_hours += draft_duration
                        current_state_is_draft = False
                    elif event['event'] == 'converted_to_draft' and not current_state_is_draft:
                        draft_start = event['created_at']
                        current_state_is_draft = True

                if current_state_is_draft and pr.merged_at:
                    final_duration = (pr.merged_at - draft_start).total_seconds() / 3600
                    draft_time_hours += final_duration

            return draft_time_hours

        except Exception as e:
            logger.warning(f"Error calculating draft time for PR #{pr.number}: {e}")
            return 0.0

    def get_ci_summary(self, pr_data_list: List[Dict]) -> Dict:
        """Generate a summary of CI data from PR data with embedded ci_data."""
        if not pr_data_list:
            return {}

        all_ci_runs = []
        pr_count = 0
        
        for pr_data in pr_data_list:
            ci_data = pr_data.get('ci_data', [])
            if ci_data:
                all_ci_runs.extend(ci_data)
                pr_count += 1

        if not all_ci_runs:
            return {}

        summary: Dict[str, Any] = {
            'total_ci_runs': len(all_ci_runs),
            'unique_prs': pr_count,
            'unique_ci_tests': len(set(run.get('ci_name', '') for run in all_ci_runs)),
        }

        states = [run.get('ci_state') for run in all_ci_runs if run.get('ci_state')]
        if states:
            state_counts = {}
            for state in states:
                state_counts[state] = state_counts.get(state, 0) + 1
            
            total_runs = len(states)
            success_count = state_counts.get('success', 0)
            failure_count = state_counts.get('failure', 0) + state_counts.get('error', 0)
            
            summary['success_rate'] = success_count / total_runs if total_runs > 0 else 0.0
            summary['failure_rate'] = failure_count / total_runs if total_runs > 0 else 0.0
            summary['state_distribution'] = state_counts

        durations = [run.get('ci_duration_seconds') for run in all_ci_runs 
                    if run.get('ci_duration_seconds') is not None]
        if durations:
            summary['avg_duration_seconds'] = sum(durations) / len(durations)
            summary['median_duration_seconds'] = sorted(durations)[len(durations) // 2]
            summary['max_duration_seconds'] = max(durations)
            summary['min_duration_seconds'] = min(durations)

        test_names = [run.get('ci_name') for run in all_ci_runs if run.get('ci_name')]
        if test_names:
            test_counts = {}
            for name in test_names:
                test_counts[name] = test_counts.get(name, 0) + 1
            
            sorted_tests = sorted(test_counts.items(), key=lambda x: x[1], reverse=True)
            summary['top_ci_tests'] = dict(sorted_tests[:10])  # Get top 10 tests

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