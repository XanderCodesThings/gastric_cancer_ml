from src.tune import tune_model
from src.constants import SEED
from src.utils import evaluate
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(random_state=SEED)

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

best_rf, best_params, best_auc = tune_model (rf, RF_PARAM_GRID)

rf_results = evaluate(best_rf)