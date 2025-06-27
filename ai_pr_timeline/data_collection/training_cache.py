"""
Training cache module for storing and retrieving PR data.
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)


class TrainingCache:
    """Manages caching of merged PR data for training purposes."""

    def __init__(self, cache_dir: str = "training_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        logger.info(f"Training cache initialized at: {self.cache_dir}")

    def _get_repo_cache_dir(self, repo_name: str) -> Path:
        """Get the cache directory for a specific repository."""
        # Replace '/' with '_' to create valid directory names
        safe_repo_name = repo_name.replace('/', '_')
        repo_dir = self.cache_dir / safe_repo_name
        repo_dir.mkdir(exist_ok=True)
        return repo_dir

    def _get_pr_cache_file(self, repo_name: str, pr_number: int) -> Path:
        """Get the cache file path for a specific PR."""
        repo_dir = self._get_repo_cache_dir(repo_name)
        return repo_dir / f"pr_{pr_number}.json"

    def _get_pr_hash(self, pr_data: Dict) -> str:
        """Generate a hash for PR data to detect changes."""
        # Use key fields that indicate if PR data has changed
        merged_at = pr_data.get('merged_at')
        key_fields = {
            'pr_number': pr_data.get('pr_number'),
            'merged_at': merged_at.isoformat() if merged_at else None,
            'title': pr_data.get('title'),
            'review_count': pr_data.get('review_count'),
            'comment_count': pr_data.get('comment_count'),
            'commit_count': pr_data.get('commit_count'),
            'files_changed': pr_data.get('files_changed'),
            'additions': pr_data.get('additions'),
            'deletions': pr_data.get('deletions')
        }
        
        # Create hash from key fields
        data_str = json.dumps(key_fields, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()

    def is_pr_cached(self, repo_name: str, pr_number: int) -> bool:
        """Check if a PR is already cached."""
        cache_file = self._get_pr_cache_file(repo_name, pr_number)
        return cache_file.exists()

    def get_cached_pr(self, repo_name: str, pr_number: int) -> Optional[Dict]:
        """Retrieve cached PR data."""
        cache_file = self._get_pr_cache_file(repo_name, pr_number)
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Convert datetime strings back to datetime objects if needed
            if 'created_at' in cached_data and isinstance(cached_data['created_at'], str):
                from datetime import datetime
                cached_data['created_at'] = datetime.fromisoformat(cached_data['created_at'].replace('Z', '+00:00'))
            
            if 'merged_at' in cached_data and isinstance(cached_data['merged_at'], str):
                from datetime import datetime
                cached_data['merged_at'] = datetime.fromisoformat(cached_data['merged_at'].replace('Z', '+00:00'))
            
            ci_count = len(cached_data.get('ci_data', []))
            logger.debug(f"📖 Loaded cache file: {cache_file.relative_to(self.cache_dir)} (PR #{pr_number}, {ci_count} CI runs)")
            return cached_data
            
        except Exception as e:
            logger.warning(f"Error reading cached PR #{pr_number} from {repo_name}: {e}")
            return None

    def cache_pr(self, repo_name: str, pr_data: Dict, ci_data: Optional[List[Dict]] = None) -> bool:
        """
        Cache PR data (and optional CI data).
        
        Args:
            repo_name: Repository name
            pr_data: PR data dictionary
            ci_data: Optional list of CI run data for this PR
            
        Returns:
            True if cached, False if error occurred
        """
        pr_number = pr_data.get('pr_number')
        if not pr_number:
            logger.warning("PR data missing pr_number, cannot cache")
            return False
        
        # Always cache if we have CI data, or if it's a merged PR
        has_ci_data = ci_data and len(ci_data) > 0
        is_merged = pr_data.get('merged_at') is not None
        
        if not has_ci_data and not is_merged:
            logger.debug(f"Skipping cache for PR #{pr_number} - not merged and no CI data")
            return False
        
        cache_file = self._get_pr_cache_file(repo_name, pr_number)
        
        try:
            # Check if we need to update the cache
            should_cache = True
            if cache_file.exists():
                existing_data = self.get_cached_pr(repo_name, pr_number)
                if existing_data:
                    existing_hash = existing_data.get('_cache_hash')
                    new_hash = self._get_pr_hash(pr_data)
                    if existing_hash == new_hash:
                        logger.debug(f"PR #{pr_number} already cached with same data")
                        should_cache = False
            
            if should_cache:
                # Check if file exists before writing (for logging)
                file_exists = cache_file.exists()
                
                # Prepare data for caching (convert datetime objects to strings)
                cache_data = pr_data.copy()
                
                # Convert datetime objects to ISO format strings
                if 'created_at' in cache_data and hasattr(cache_data['created_at'], 'isoformat'):
                    cache_data['created_at'] = cache_data['created_at'].isoformat()
                
                if 'merged_at' in cache_data and hasattr(cache_data['merged_at'], 'isoformat'):
                    cache_data['merged_at'] = cache_data['merged_at'].isoformat()
                
                # Add CI data if provided
                if ci_data:
                    # Convert datetime objects in CI data to ISO format strings
                    ci_data_serializable = []
                    for ci_run in ci_data:
                        ci_run_copy = ci_run.copy()
                        for date_field in ['ci_created_at', 'ci_updated_at', 'pr_created_at', 'pr_merged_at']:
                            if date_field in ci_run_copy and hasattr(ci_run_copy[date_field], 'isoformat'):
                                ci_run_copy[date_field] = ci_run_copy[date_field].isoformat()
                        ci_data_serializable.append(ci_run_copy)
                    cache_data['ci_data'] = ci_data_serializable
                
                # Add cache metadata
                cache_data['_cache_hash'] = self._get_pr_hash(pr_data)
                cache_data['_cached_at'] = str(datetime.now())
                
                # Write to cache file
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, indent=2, default=str)
                
                # Log the file operation with details
                ci_count = len(ci_data) if ci_data else 0
                action = "Updated" if file_exists else "Created"
                cache_type = "merged PR" if is_merged else "CI data"
                logger.info(f"📁 {action} cache file: {cache_file.relative_to(self.cache_dir)} (PR #{pr_number}, {ci_count} CI runs, {cache_type})")
                logger.debug(f"Cached PR #{pr_number} from {repo_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error caching PR #{pr_number} from {repo_name}: {e}")
            return False

    def get_cached_prs_for_repo(self, repo_name: str) -> List[Dict]:
        """Get all cached PRs for a repository."""
        repo_dir = self._get_repo_cache_dir(repo_name)
        cached_prs = []
        
        if not repo_dir.exists():
            return cached_prs
        
        for cache_file in repo_dir.glob("pr_*.json"):
            try:
                pr_number = int(cache_file.stem.split('_')[1])
                pr_data = self.get_cached_pr(repo_name, pr_number)
                if pr_data:
                    cached_prs.append(pr_data)
            except Exception as e:
                logger.warning(f"Error reading cache file {cache_file}: {e}")
                continue
        
        total_ci_runs = sum(len(pr.get('ci_data', [])) for pr in cached_prs)
        logger.info(f"📚 Retrieved {len(cached_prs)} cached PRs for {repo_name} (total: {total_ci_runs} CI runs)")
        return cached_prs

    def get_cached_pr_numbers(self, repo_name: str) -> Set[int]:
        """Get set of cached PR numbers for a repository."""
        repo_dir = self._get_repo_cache_dir(repo_name)
        cached_numbers = set()
        
        if not repo_dir.exists():
            return cached_numbers
        
        for cache_file in repo_dir.glob("pr_*.json"):
            try:
                pr_number = int(cache_file.stem.split('_')[1])
                cached_numbers.add(pr_number)
            except ValueError:
                continue
        
        return cached_numbers

    def get_cached_pr_numbers_with_ci(self, repo_name: str) -> Set[int]:
        """Get set of cached PR numbers that have CI data for a repository."""
        repo_dir = self._get_repo_cache_dir(repo_name)
        cached_numbers_with_ci = set()
        
        if not repo_dir.exists():
            return cached_numbers_with_ci
        
        for cache_file in repo_dir.glob("pr_*.json"):
            try:
                pr_number = int(cache_file.stem.split('_')[1])
                pr_data = self.get_cached_pr(repo_name, pr_number)
                if pr_data and pr_data.get('ci_data'):
                    cached_numbers_with_ci.add(pr_number)
            except ValueError:
                continue
        
        return cached_numbers_with_ci

    def get_all_cached_ci_data(self, repo_name: str) -> List[Dict]:
        """Get all cached CI data for a repository."""
        all_ci_data = []
        cached_prs = self.get_cached_prs_for_repo(repo_name)
        
        for pr_data in cached_prs:
            ci_data = pr_data.get('ci_data', [])
            if ci_data:
                all_ci_data.extend(ci_data)
        
        logger.info(f"🔧 Retrieved {len(all_ci_data)} cached CI runs for {repo_name}")
        return all_ci_data

    def clear_repo_cache(self, repo_name: str) -> int:
        """Clear all cached data for a repository."""
        repo_dir = self._get_repo_cache_dir(repo_name)
        count = 0
        
        if repo_dir.exists():
            for cache_file in repo_dir.glob("pr_*.json"):
                try:
                    cache_file.unlink()
                    count += 1
                except Exception as e:
                    logger.warning(f"Error deleting cache file {cache_file}: {e}")
        
        logger.info(f"Cleared {count} cached PRs for {repo_name}")
        return count

    def get_cache_stats(self) -> Dict:
        """Get statistics about the training cache."""
        stats = {
            'total_repos': 0,
            'total_prs': 0,
            'repos': {}
        }
        
        if not self.cache_dir.exists():
            return stats
        
        for repo_dir in self.cache_dir.iterdir():
            if repo_dir.is_dir():
                repo_name = repo_dir.name.replace('_', '/')
                pr_count = len(list(repo_dir.glob("pr_*.json")))
                
                stats['repos'][repo_name] = pr_count
                stats['total_repos'] += 1
                stats['total_prs'] += pr_count
        
        return stats 