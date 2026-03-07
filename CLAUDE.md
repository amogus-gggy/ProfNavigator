# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ProfNavigator** — a career orientation system with ML classification. Users answer a survey (15 random questions), and the system predicts their best-fit professional sphere from 10 categories using a trained classifier.

## Commands

### Run the server
```bash
uvicorn main:app --reload
```
Access at http://127.0.0.1:8000

### Install dependencies
```bash
pip install -r requirements.txt
# Optional advanced ML libraries:
pip install lightgbm xgboost catboost
```

### Train a model
```bash
# Standard training (recommended)
python trainer.py --model-type random_forest

# With hyperparameter optimization (takes many hours)
python trainer.py --model-type random_forest --optimize --n-iter 200

# With probability calibration
python trainer.py --model-type extra_trees --calibrate

# Generate new synthetic dataset first, then train
python data_gen.py
python trainer.py --model-type random_forest
```

## Architecture

The app is a FastAPI server with a single-page HTML frontend.

**Request flow:**
1. `GET /questions` — returns N random questions from `questions.json` with shuffled options
2. User answers in `static/index.html`
3. `POST /submit` — maps `(question_id, option_id)` pairs to categories, builds a feature vector of 10 category counts, passes to `SurveyModel.predict()`, saves to `responses.json`

**Key files:**
- `main.py` — FastAPI app, endpoints, response persistence
- `model.py` — `SurveyModel` class: loads `model_artifact.pkl` on startup; falls back to a simple `DecisionTreeClassifier` if the artifact is missing
- `trainer.py` — standalone script to train and save `model_artifact.pkl`; supports 10 model types with optional `RandomizedSearchCV` and probability calibration
- `data_gen.py` — generates synthetic `dataset.json` with per-category profiles
- `questions.json` — survey questions; each option has a `category` field mapping to one of the 10 categories
- `dataset.json` — training data: list of `{features: {category: count}, label: str}`
- `responses.json` — accumulated user responses (same format as `dataset.json`), auto-created at runtime

**Model artifact format** (`model_artifact.pkl`):
```python
{
    "model": <sklearn-compatible classifier>,
    "label_encoder": <LabelEncoder>,
    "categories": [...],  # list of 10 category strings
    "metrics": {"validation_accuracy": float, "cv_mean": float, "cv_std": float, ...}
}
```

**10 professional categories:** `analytical`, `social`, `creative`, `managerial`, `practical`, `research`, `technical`, `artistic`, `entrepreneurial`, `scientific`

## Stable Models

Only `random_forest`, `lightgbm`, and `xgboost` are production-stable. Other model types (`extra_trees`, `gradient_boosting`, `neural_network`, `voting`, `stacking`) may produce unstable results.

## Retraining on Accumulated Data

To retrain on user responses collected in `responses.json`:
1. Copy the `samples` array from `responses.json` into `dataset.json`
2. Run `python trainer.py --model-type random_forest`

The `/api/retrain` endpoint exists in `main.py` but is commented out (no auth implemented).
