# AI PR Timeline Predictor

A machine learning tool that predicts how long GitHub pull requests will take to merge based on historical data and PR characteristics.

## Features

- **Smart Data Collection**: Efficiently collects PR data from GitHub repositories with intelligent caching
- **Advanced Feature Engineering**: Extracts and balances structural and text features for optimal predictions
- **Multiple ML Models**: Supports Random Forest, XGBoost, and LightGBM algorithms
- **Training Cache System**: Reduces API usage by caching processed PR data locally
- **API Call Monitoring**: Tracks and logs all GitHub API calls with detailed statistics
- **Flexible Training Options**: Control data collection with granular parameters
- **Batch Predictions**: Predict timelines for multiple PRs across repositories
- **Feature Balancing**: Intelligently weights text vs structural features for better accuracy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/ai-pr-timeline.git
cd ai-pr-timeline
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your GitHub token:
```bash
export GITHUB_TOKEN="your_github_personal_access_token"
```

## Quick Start

### 1. Train a Model

Train a model on a specific repository:

```bash
python commands/train_model.py \
  --repo "microsoft/vscode" \
  --model-type random_forest \
  --max-prs 500 \
  --max-new-prs 100
```

Key parameters:
- `--max-prs`: Total PRs to use for training (including cached)
- `--max-new-prs`: Maximum new PRs to fetch from API (0 = use only cache)
- `--text-weight`: Weight for text features (0.0-1.0, default: 0.3)
- `--max-text-features`: Number of text features to extract (default: 50)

### 2. Make Predictions

Predict timeline for a specific PR:

```bash
python commands/predict_pr.py \
  --repo "microsoft/vscode" \
  --pr-number 12345 \
  --model "your_model.pkl"
```

### 3. Batch Predictions

Create a file with repository names (one per line):
```bash
echo "microsoft/vscode" > my_repos.txt
echo "facebook/react" >> my_repos.txt
```

Run batch predictions:
```bash
python commands/batch_predict.py \
  --repos-file my_repos.txt \
  --model "your_model.pkl" \
  --output results.csv
```

## Core Components

### Enhanced Data Collection (`GitHubDataCollector`)

Efficiently collects both PR and CI data with:
- **Single API Pass**: Collects both PR and CI data in one efficient pass
- **Selective Collection**: Choose to collect PR data, CI data, or both
- **Training Cache**: Stores processed data to reduce API calls
- **Bot Detection**: Automatically filters out bot-authored PRs
- **Draft Time Calculation**: Excludes draft periods from merge time calculations
- **API Call Tracking**: Monitors and logs all GitHub API interactions
- **Rate Limit Handling**: Automatically handles GitHub API rate limits
- **Backward Compatibility**: Drop-in replacement for old collectors

### Feature Engineering (`FeatureEngineer`)

Extracts and processes features:

**Structural Features** (16 features):
- Code metrics: files changed, additions, deletions, lines changed
- Activity metrics: review count, comment count, commit count
- Ratios: addition ratio, files per addition, reviews per commit
- Time features: created hour/day, business hours, weekend flags
- PR metadata: draft status, author association

**Text Features** (up to 50 features):
- TF-IDF vectors from PR titles and descriptions
- Advanced preprocessing with stop words and n-grams
- Balanced weighting to prevent text feature dominance

**Feature Balancing**:
- Text features weighted by configurable factor (default: 0.3)
- Prevents text features from overwhelming structural features
- Maintains interpretability while leveraging text signals

### Model Training (`ModelTrainer`)

Supports multiple algorithms:
- **Random Forest**: Robust ensemble method with feature importance
- **XGBoost**: Gradient boosting with advanced regularization
- **LightGBM**: Fast gradient boosting optimized for large datasets

Features:
- Hyperparameter tuning with GridSearchCV
- Cross-validation for model validation
- Comprehensive evaluation metrics (MAE, RMSE, R², MAPE)
- Feature importance analysis
- Model persistence with metadata

### Training Cache System

Intelligent caching reduces API usage:
- **File Structure**: `training_cache/{owner}_{repo}/pr_{number}.json`
- **Cache Validation**: Hash-based integrity checking
- **Automatic Updates**: Refreshes stale cache entries
- **Repository Management**: Per-repo cache organization
- **Statistics**: Detailed cache usage reporting

## Advanced Usage

### Training with Cache Control

```bash
# Use only cached data (no API calls for new PRs)
python commands/train_model.py \
  --repo "owner/repo" \
  --max-prs 1000 \
  --max-new-prs 0

# Fetch up to 50 new PRs from API, use rest from cache
python commands/train_model.py \
  --repo "owner/repo" \
  --max-prs 1000 \
  --max-new-prs 50

# Custom feature balancing
python commands/train_model.py \
  --repo "owner/repo" \
  --text-weight 0.1 \
  --max-text-features 20
```

### Model Types and Hyperparameter Tuning

```bash
# Train XGBoost with hyperparameter tuning
python commands/train_model.py \
  --repo "owner/repo" \
  --model-type xgboost \
  --tune-hyperparams

# Train LightGBM model
python commands/train_model.py \
  --repo "owner/repo" \
  --model-type lightgbm
```

### Custom Output Filenames

```bash
# Specify custom model filename
python commands/train_model.py \
  --repo "owner/repo" \
  --output-filename "my_custom_model.pkl"

# Auto-generated timestamped filename (default)
# Creates: owner_repo_random_forest_model_20241225_143022.pkl
```

## Configuration

The system uses a `Config` class with these key settings:

```python
# Feature Engineering
include_text_features: bool = True
max_text_features: int = 50  # Reduced for better balance
text_feature_weight: float = 0.3  # Weight for text features

# Data Collection  
max_prs_per_repo: int = 1000
min_data_points: int = 50

# Model Training
model_type: str = "random_forest"
test_size: float = 0.2
random_state: int = 42

# Directories
training_cache_dir: str = "training_cache"
model_dir: str = "models"
data_dir: str = "data"
```

## API Usage Monitoring

The system tracks all GitHub API calls:

```
API Call #1: GET /repos/microsoft/vscode - Getting repository info
API Call #2: GET /repos/microsoft/vscode/pulls - Getting closed PRs
API Call #3: GET /repos/microsoft/vscode/pulls/12345/reviews - PR #12345 reviews
...
Total API calls made: 156
```

This helps you:
- Monitor API quota usage
- Optimize data collection strategies
- Debug collection issues
- Plan cache usage effectively

## Performance Optimization

### Feature Balancing Results

Before optimization:
- 16 structural + 1000 text features = 1016 total
- Text-to-structural ratio: 62.5:1
- Text features dominated model decisions

After optimization:  
- 16 structural + 50 text features = 66 total
- Effective ratio after weighting: 0.9:1
- Structural features maintain dominance

### Cache Efficiency

Typical cache hit rates:
- **First run**: 0% cache hits, ~200-400 API calls
- **Subsequent runs**: 80-95% cache hits, ~10-50 API calls
- **Cache-only runs** (`--max-new-prs 0`): 100% cache hits, ~2 API calls

## Model Performance

Typical performance metrics on well-represented repositories:

- **MAE (Mean Absolute Error)**: 8-24 hours
- **RMSE (Root Mean Squared Error)**: 15-45 hours  
- **R² Score**: 0.3-0.7
- **MAPE (Mean Absolute Percentage Error)**: 25-60%

Performance varies significantly based on:
- Repository characteristics and consistency
- Amount of training data available
- Feature quality and completeness
- Model type and hyperparameters

## File Structure

```
ai-pr-timeline/
├── ai_pr_timeline/           # Main package
│   ├── __init__.py
│   ├── config.py                    # Configuration settings
│   ├── data_collector.py           # Enhanced PR and CI data collection
│   ├── training_cache.py           # Cache management system
│   ├── utils.py                    # Utility functions
│   ├── merge_time/                 # PR merge time prediction
│   │   ├── __init__.py
│   │   ├── feature_engineer.py    # PR feature extraction
│   │   ├── model_trainer.py       # PR model training
│   │   └── predictor.py           # PR prediction interface
│   └── ci/                        # CI prediction
│       ├── __init__.py
│       ├── feature_engineer.py    # CI feature extraction
│       ├── model_trainer.py       # CI model training
│       └── predictor.py           # CI prediction interface
├── commands/                 # Command-line tools
│   ├── train_model.py       # Model training command
│   ├── predict_pr.py        # Single PR prediction
│   └── batch_predict.py     # Batch prediction tool
├── training_cache/          # Cached PR data (auto-created)
├── models/                  # Trained models (auto-created)
├── data/                    # Raw data exports (auto-created)
└── tests/                   # Test suite
```

## Troubleshooting

### Common Issues

**API Rate Limits**:
- Use `--max-new-prs` to limit API calls
- Leverage cache with `--max-new-prs 0`
- Spread data collection across multiple sessions

**Insufficient Training Data**:
- Increase `--max-prs` parameter
- Use repositories with more historical PRs
- Combine multiple repositories for training

**Poor Model Performance**:
- Ensure sufficient training data (>100 PRs recommended)
- Try different model types (`--model-type`)
- Adjust feature balancing (`--text-weight`)
- Enable hyperparameter tuning (`--tune-hyperparams`)

**Cache Issues**:
- Check `training_cache/` directory permissions
- Clear cache for specific repo if corrupted
- Verify GitHub token has proper repository access

### Debug Mode

Enable detailed logging:
```bash
export PYTHONPATH=.
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
# Run your command here
"
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`python -m pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with scikit-learn, XGBoost, and LightGBM
- Uses PyGithub for GitHub API integration
- Inspired by the need for better PR workflow planning
