from sklearn.model_selection import StratifiedKFold, cross_validation
from sklearn.metrics import make_scorer, recall_score
from pathlib import Path
# Constant seed for reproduceability
SEED = 0

# The same 5-fold split for every model. 
# Stratified keeps the cancer/control ratio in each fold. 
# Shuffle with a fixed seed makes the folds reproducible.
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Study Metrics.
# Sensitivity = recall on cancer (label 1).
# Specificity = recall on control (label 0).
SCORING = {
    "AUC": "roc_auc",
    "Sensitivity": "recall",
    "Specificity": make_scorer(recall_score, pos_label=0),
    "Accuracy": "accuracy",
    "F1": "f1",
}

# Constant file path to access data from any location
DATA_PATH = Path(__file__).resolve().parent / "data" / "gastric.csv"