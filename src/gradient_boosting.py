from src.tune import tune_model
from src.constants import SEED
from src.utils import evaluate
from sklearn.ensemble import HistGradientBoostingClassifier

RF = HistGradientBoostingClassifier(random_state=SEED)

GBDT_PARAM_GRID = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_iter": [100, 200, 300],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [20, 30, 40]
}

best_gbdt, best_params, best_auc = tune_model(gbdt, GBDT_PARAM_GRID)

gbdt_results = evaluate(best_gbdt)