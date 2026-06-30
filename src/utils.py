from constants import DATA_PATH, CV, SEED, SCORING
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
    df = pd.read_csv(DATA_PATH).drop_duplicates
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
    return pd.DataFrame({name: evaluate(clf) for name, clf in models.items()}).T
