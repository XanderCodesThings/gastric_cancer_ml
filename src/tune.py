from src.utils import load_data
from src.constants import CV
from sklearn.model_selection import GridSearchCV

def tune_model(model, param_grid, scoring="roc_auc"):
    """
    Tune a scikit-learn model using GridSearchCV.

    Parameters
    ----------
    model : estimator
        Untrained scikit-learn estimator.

    param_grid : dict
        Hyperparameter search space.

    scoring : str
        Evaluation metric used during tuning.

    Returns
    -------
    best_model
        Best fitted estimator.

    best_params : dict
        Best hyperparameters.

    best_score : float
        Mean CV score of best estimator.
    """

    X, y = load_data()

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=CV,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X, y)

    return (
        search.best_estimator_,
        search.best_params_,
        search.best_score_,
    )