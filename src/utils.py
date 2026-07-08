from src.constants import DATA_PATH, CV, SCORING, METRICS_DIR, MODELS_DIR, METADATA_DIR
import json
import joblib
import pandas as pd
from sklearn.model_selection import cross_validate

def load_data():
    """
    Load the gastric cancer dataset.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target labels.
    """
    df = pd.read_csv(DATA_PATH).drop_duplicates()
    return df.drop(columns="label"), df["label"]

def evaluate(model):
    """Cross-validate one model with the shared split and metrics.
    Returns a dict of mean scores (rounded to 3) across the 5 folds.
    """
    results = {}

    X, y = load_data()
    scores = cross_validate(model, X, y, cv=CV, scoring=SCORING)

    for metric in SCORING:
        results[metric] = round(
            scores[f"test_{metric}"].mean(),
            3
        )

    return results


def compare(models):
    """Evaluate a dict of {name: model} and return a metrics table (one row per model) as a DataFrame."""
    results = {
        name: evaluate(model)
        for name, model in models.items()
    }

    return pd.DataFrame(results).T

def save_model(model, filename):
    joblib.dump(model, MODELS_DIR / filename)

def save_training_results(results):
    with open(
        METADATA_DIR / "training_results.json",
        "w",
    ) as f:
        json.dump(results, f, indent=4)

def save_metrics(df):
    df.to_csv(
        METRICS_DIR / "model_metrics.csv"
    )