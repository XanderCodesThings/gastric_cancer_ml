from tune import tune_model
from constants import SEED
from utils import evaluate
from sklearn.ensemble import GradientBoostingClassifier

gbdt = GradientBoostingClassifier(random_state=SEED)

GBDT_PARAM_GRID = {
    "histgradientboostingclassifier__learning_rate": [0.01, 0.05, 0.1],
    "histgradientboostingclassifier__max_iter": [100, 200, 300],
    "histgradientboostingclassifier__max_leaf_nodes": [15, 31, 63],
    "histgradientboostingclassifier__min_samples_leaf": [20, 30, 40]
}

best_gbdt, best_params, best_auc = tune_model(gbdt, GBDT_PARAM_GRID)

gbdt_results = evaluate(best_gbdt)