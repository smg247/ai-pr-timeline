# Training Cache

The AI PR Timeline tool includes a training cache system to reduce GitHub API usage during model training by storing merged PR data locally.

## How It Works

- **Automatic Caching**: When collecting PR data for training, merged PRs are automatically cached to the `training_cache/` directory
- **Cache Structure**: Data is organized as `training_cache/{owner}_{repo}/pr_{number}.json`
- **Intelligent Retrieval**: On subsequent training runs, cached data is used instead of making API calls
- **Merged PRs Only**: Only merged PRs are cached since they have complete data needed for training

## Benefits

- **Reduced API Usage**: Dramatically reduces GitHub API token consumption
- **Faster Training**: Subsequent training runs are much faster
- **Offline Training**: Can train models using cached data without internet connection
- **Data Persistence**: Training data is preserved between sessions

## File Structure

```
training_cache/
├── microsoft_vscode/
│   ├── pr_12345.json
│   ├── pr_12346.json
│   └── ...
├── facebook_react/
│   ├── pr_54321.json
│   └── ...
└── ...
```

## Cache Management

### View Cache Stats
```python
from ai_pr_timeline import TrainingCache

cache = TrainingCache()
stats = cache.get_cache_stats()
print(f"Total cached PRs: {stats['total_prs']}")
```

### Clear Repository Cache
```python
cache.clear_repo_cache("owner/repo")
```

### Check if PR is Cached
```python
is_cached = cache.is_pr_cached("owner/repo", 12345)
```

## Usage Examples

### Demo Script
```bash
# Show cache functionality
python examples/cache_demo.py --repo "microsoft/vscode" --max-prs 50

# Clear cache and rebuild
python examples/cache_demo.py --repo "microsoft/vscode" --clear-cache --max-prs 50
```

### Training with Cache
```bash
# First run - will cache PRs as they're fetched
python examples/train_model.py --repo "microsoft/vscode" --max-prs 500

# Second run - will use cached data (much faster)
python examples/train_model.py --repo "microsoft/vscode" --max-prs 500
```

## Configuration

The cache directory can be configured in the Config class:

```python
from ai_pr_timeline import Config

config = Config()
config.training_cache_dir = "my_custom_cache"  # Default: "training_cache"
```

## Cache Data Format

Each cached PR file contains:

```json
{
  "pr_number": 12345,
  "title": "Fix bug in feature X",
  "body": "This PR fixes...",
  "review_count": 3,
  "comment_count": 5,
  "commit_count": 2,
  "files_changed": 4,
  "additions": 150,
  "deletions": 30,
  "created_hour": 14,
  "created_day": 2,
  "author_association": "MEMBER",
  "is_draft": false,
  "created_at": "2023-01-15T14:30:00Z",
  "merged_at": "2023-01-16T10:15:00Z",
  "repository": "microsoft/vscode",
  "merge_time_hours": 19.75,
  "_cache_hash": "abc123...",
  "_cached_at": "2023-01-20 15:30:00"
}
```

## Important Notes

- **Git Ignored**: The `training_cache/` directory is automatically ignored by git
- **Merged PRs Only**: Only merged PRs are cached to ensure complete training data
- **Hash Validation**: Cached data includes hash validation to detect changes
- **Automatic Updates**: Cache is automatically updated if PR data changes
- **Thread Safe**: Cache operations are designed to be safe for concurrent access

## API Reference

### TrainingCache Class

```python
class TrainingCache:
    def __init__(self, cache_dir: str = "training_cache")
    def is_pr_cached(self, repo_name: str, pr_number: int) -> bool
    def get_cached_pr(self, repo_name: str, pr_number: int) -> Optional[Dict]
    def cache_pr(self, repo_name: str, pr_data: Dict) -> bool
    def get_cached_prs_for_repo(self, repo_name: str) -> List[Dict]
    def get_cached_pr_numbers(self, repo_name: str) -> Set[int]
    def clear_repo_cache(self, repo_name: str) -> int
    def get_cache_stats(self) -> Dict
```

The training cache is automatically integrated into the data collection process and requires no additional configuration for basic usage. 