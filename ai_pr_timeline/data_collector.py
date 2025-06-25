"""
Data collection module for fetching PR data from GitHub API.
"""

import time
import logging
from typing import List, Dict, Optional
import pandas as pd
import requests
from github import Github
from github.GithubException import RateLimitExceededException

from .config import Config, DEFAULT_CONFIG
from .training_cache import TrainingCache

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

    def collect_pr_data(self, repo_name: str, limit: Optional[int] = None, 
                       max_new_prs: Optional[int] = None) -> pd.DataFrame:
        """
        Collect PR data from a specific repository, using cache when available.

        Args:
            repo_name: Repository name in format 'owner/repo'
            limit: Maximum number of PRs to collect (total, including cached)
            max_new_prs: Maximum number of new PRs to fetch from API (must be <= limit)

        Returns:
            DataFrame with PR data
        """
        logger.info(f"Collecting PR data from {repo_name}")
        
        # Check cache first
        cached_prs = self.training_cache.get_cached_prs_for_repo(repo_name)
        cached_pr_numbers = self.training_cache.get_cached_pr_numbers(repo_name)
        
        if cached_prs:
            logger.info(f"Found {len(cached_prs)} cached PRs for {repo_name}")
        
        pr_data = cached_prs.copy()  # Start with cached data
        count = len(pr_data)
        max_prs = limit or self.config.max_prs_per_repo
        max_new = max_new_prs if max_new_prs is not None else max_prs
        
        # Validate max_new_prs parameter
        if max_new > max_prs:
            raise ValueError(f"max_new_prs ({max_new}) cannot be greater than limit ({max_prs})")
        
        # If we have enough cached data, return it
        if count >= max_prs:
            logger.info(f"Using {max_prs} cached PRs from {repo_name} (no API calls needed)")
            df = pd.DataFrame(pr_data[:max_prs])
            return df

        try:
            self._log_api_call(f"GET /repos/{repo_name}", f"Getting repository info")
            repo = self.github.get_repo(repo_name)
            
            self._log_api_call(f"GET /repos/{repo_name}/pulls", f"Getting closed PRs")
            prs = repo.get_pulls(state='closed', sort='updated', direction='desc')

            new_prs_processed = 0
            for pr in prs:
                if count >= max_prs or new_prs_processed >= max_new:
                    break

                try:
                    # Skip if already cached
                    if pr.number in cached_pr_numbers:
                        continue
                    
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
                            new_prs_processed += 1

                            # Cache the PR data
                            self.training_cache.cache_pr(repo_name, data)

                            # Log details about each PR processed
                            merge_time_hours = data.get('merge_time_hours', 0)
                            merge_time_days = merge_time_hours / 24
                            pr_title = pr.title[:60] + "..." if len(pr.title) > 60 else pr.title

                            # Calculate total time for comparison
                            total_time_hours = (pr.merged_at - pr.created_at).total_seconds() / 3600
                            draft_time_hours = total_time_hours - merge_time_hours

                            draft_info = (f" (excl. {draft_time_hours:.1f}h draft)"
                                        if draft_time_hours > 0.1 else "")

                            logger.info(
                                f"Processed PR #{pr.number}: '{pr_title}' | "
                                f"Merge time: {merge_time_hours:.1f}h ({merge_time_days:.1f}d)"
                                f"{draft_info} | "
                                f"Files: {pr.changed_files}, +{pr.additions}/-{pr.deletions}"
                            )

                            if count % 50 == 0:
                                logger.info(f"--- Collected {count} PRs so far ({new_prs_processed} new, {len(cached_prs)} cached) ---")

                except Exception as e:
                    logger.warning(f"Error processing PR #{pr.number}: {e}")
                    continue

        except RateLimitExceededException:
            logger.warning("GitHub API rate limit exceeded. Waiting...")
            time.sleep(60)
            return self.collect_pr_data(repo_name, limit, max_new_prs)

        df = pd.DataFrame(pr_data)
        if max_new_prs and max_new_prs < max_prs:
            logger.info(f"Collected {len(df)} PRs from {repo_name} ({new_prs_processed} newly processed, {len(cached_prs)} from cache) - limited to {max_new} new PRs")
        else:
            logger.info(f"Collected {len(df)} PRs from {repo_name} ({new_prs_processed} newly processed, {len(cached_prs)} from cache)")
        logger.info(f"Total API calls made in this session: {self.api_call_count}")
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
                'snyk',
                'renovate-bot',
                'dependabot-preview',
                'allcontributors',
                'imgbot',
                'semantic-release-bot'
            ]

            return any(pattern in username for pattern in bot_patterns)

        except Exception as e:
            logger.warning(f"Error checking if PR author is bot: {e}")
            return False

    def _calculate_draft_time(self, pr) -> float:
        """
        Calculate total time the PR spent in draft state.

        Args:
            pr: GitHub PullRequest object

        Returns:
            Total draft time in hours
        """
        try:
            # Get timeline events
            self._log_api_call(f"GET /repos/{pr.base.repo.full_name}/issues/{pr.number}/events", f"PR #{pr.number} timeline events")
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

    def collect_multiple_repos(self, repo_names: List[str],
                             limit_per_repo: Optional[int] = None,
                             max_new_prs_per_repo: Optional[int] = None) -> pd.DataFrame:
        """
        Collect PR data from multiple repositories.

        Args:
            repo_names: List of repository names in format 'owner/repo'
            limit_per_repo: Maximum number of PRs to collect per repository
            max_new_prs_per_repo: Maximum number of new PRs to fetch from API per repository

        Returns:
            Combined DataFrame with PR data from all repositories
        """
        all_data = []

        for repo_name in repo_names:
            try:
                repo_data = self.collect_pr_data(repo_name, limit_per_repo, max_new_prs_per_repo)
                all_data.append(repo_data)
            except Exception as e:
                logger.error(f"Failed to collect data from {repo_name}: {e}")
                continue

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Collected total of {len(combined_df)} PRs from {len(all_data)} repositories")
            return combined_df
        else:
            logger.warning("No data collected from any repository")
            return pd.DataFrame()

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
