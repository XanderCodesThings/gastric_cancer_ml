from src.constants import SEED
from sklearn.ensemble import HistGradientBoostingClassifier

GBDT = HistGradientBoostingClassifier(random_state=SEED)

GBDT_PARAM_GRID = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_iter": [100, 200, 300],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [20, 30, 40]
}