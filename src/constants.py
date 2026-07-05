from sklearn.model_selection import StratifiedKFold
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

# Constant file paths for quick access to resources
# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Project Directories
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures" # Upload figures: plt.savefig(FIGURES_DIR / "example.png")
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# File path to gastric.csv
DATA_PATH = DATA_DIR / "gastric.csv"

# Artifact Directories
METRICS_DIR = ARTIFACTS_DIR / "metrics"
MODELS_DIR = ARTIFACTS_DIR / "models"
PARAMETERS = ARTIFACTS_DIR / "parameters"