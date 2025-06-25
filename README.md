# AI PR Timeline Prediction Plugin

A Python plugin for predicting pull request merge times using machine learning and historical GitHub data.

## Overview

This plugin uses machine learning to predict how long a pull request will take to merge based on historical data from GitHub repositories. It analyzes various factors such as:

- PR size (lines changed, files modified)
- Team activity (reviews, comments, commits)
- Timing patterns (creation time, day of week)
- Author information (association with project)
- Text analysis of PR title and description

## Features

- **Multiple ML Models**: Support for Random Forest, XGBoost, and LightGBM
- **GitHub API Integration**: Automatic data collection from GitHub repositories  
- **Feature Engineering**: Smart extraction of predictive features from PR data
- **Batch Processing**: Train on multiple repositories and predict for multiple PRs
- **Text Analysis**: Optional NLP features from PR titles and descriptions
- **Comprehensive Metrics**: Detailed model evaluation and feature importance
- **Easy Integration**: Simple API for integration into other tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/ai-pr-timeline.git
cd ai-pr-timeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your GitHub token:
```bash
export GITHUB_TOKEN="your_github_personal_access_token"
```

Or create a `.env` file:
```bash
cp .env.example .env
# Edit .env and add your GitHub token
```

## Quick Start

### 1. Train a Model

Train on a single repository:
```bash
python examples/train_model.py --repo "microsoft/vscode" --model-type random_forest
```

Train on multiple repositories:
```python
from ai_pr_timeline import PRTimelinePredictor, Config

config = Config()
config.github_token = "your_token"

predictor = PRTimelinePredictor(config)
results = predictor.train_on_repositories([
    "microsoft/vscode",
    "facebook/react", 
    "google/tensorflow"
])

print(f"Model R² Score: {results['metrics']['r2']:.3f}")
print(f"Mean Absolute Error: {results['metrics']['mae']:.1f} hours")
```

### 2. Make Predictions

Predict for a specific PR:
```bash
python examples/predict_pr.py --repo "microsoft/vscode" --pr-number 123 --model "your_model.pkl"
```

Predict programmatically:
```python
from ai_pr_timeline import PRTimelinePredictor

predictor = PRTimelinePredictor()
predictor.load_trained_model("your_model.pkl")

result = predictor.predict_from_github_pr("microsoft/vscode", 123)
print(f"Predicted timeline: {result['predicted_hours']:.1f} hours")
print(f"Category: {result['time_category']}")
```

### 3. Batch Processing

Process multiple repositories:
```bash
python examples/batch_predict.py --repos-file examples/repos.txt --model "your_model.pkl" --output results.csv
```

## API Reference

### PRTimelinePredictor

Main interface for training and prediction.

#### Methods

- `train_on_repository(repo_name, save_model=True)` - Train on single repository
- `train_on_repositories(repo_names, save_model=True)` - Train on multiple repositories  
- `train_on_data(df, save_model=True)` - Train on provided DataFrame
- `load_trained_model(filename)` - Load a pre-trained model
- `predict_pr_timeline(pr_data)` - Predict from PR data dictionary
- `predict_from_github_pr(repo_name, pr_number)` - Predict directly from GitHub
- `get_model_info()` - Get model information and metrics
- `benchmark_predictions(test_data)` - Evaluate model on test data

### Configuration

Customize behavior through the `Config` class:

```python
from ai_pr_timeline import Config

config = Config()
config.model_type = "xgboost"  # or "random_forest", "lightgbm"
config.max_prs_per_repo = 500
config.include_text_features = True
config.max_text_features = 1000
```

## Model Features

The plugin automatically extracts these features from PR data:

### Basic Features
- Review count, comment count, commit count
- Files changed, lines added/deleted
- Creation hour and day of week
- Author association (member, contributor, etc.)
- Draft status

### Derived Features  
- Total changes (additions + deletions)
- Change ratio (additions / deletions)
- Files per commit
- Changes per file
- Weekend creation flag
- Business hours creation flag

### Text Features (Optional)
- TF-IDF vectors from PR title and description
- Cleaned text without code blocks and URLs
- N-gram analysis (1-gram and 2-gram)

### Feature Balancing

To ensure structural features (like code changes, reviews) don't get overwhelmed by text features, the system includes intelligent feature balancing:

- **Reduced Text Features**: Default 50 text features (vs 1000 previously)
- **Feature Weighting**: Text features weighted at 30% of structural features
- **Smart TF-IDF**: Filters common/rare terms with `min_df=2`, `max_df=0.8`
- **Sublinear Scaling**: Reduces impact of high-frequency terms

Configure feature balancing:
```bash
# Conservative text influence
python examples/train_model.py --repo "owner/repo" --text-weight 0.1 --max-text-features 20

# Disable text features entirely  
python examples/train_model.py --repo "owner/repo" --text-weight 0.0

# View feature balance demonstration
python examples/feature_balance_demo.py
```

## Performance

Typical model performance on well-established repositories:

- **Mean Absolute Error**: 8-24 hours
- **R² Score**: 0.3-0.7 
- **Accuracy within 1 day**: 60-80%
- **Accuracy within 1 week**: 85-95%

Performance varies significantly based on:
- Repository size and activity
- Team processes and consistency  
- Amount of training data
- Feature engineering quality

## Examples

See the `examples/` directory for complete usage examples:

- `train_model.py` - Train a model on repository data
- `predict_pr.py` - Make predictions for specific PRs
- `batch_predict.py` - Batch processing for multiple PRs
- `repos.txt` - Example repository list

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Limitations

- Requires significant historical data (50+ merged PRs minimum)
- Performance varies by repository and team practices
- Predictions are estimates with confidence intervals
- GitHub API rate limiting may slow data collection
- Text features require additional processing time

## Roadmap

- [ ] Support for GitLab and other platforms
- [ ] Real-time model updates
- [ ] Integration with popular CI/CD tools
- [ ] Web dashboard for visualization
- [ ] Advanced ensemble models
- [ ] Automated hyperparameter tuning
