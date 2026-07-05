from src.constants import SEED
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(random_state=SEED)

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}