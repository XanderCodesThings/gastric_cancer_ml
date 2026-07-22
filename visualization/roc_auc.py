# Visualization of ROC Curves for both models
import joblib
import matplotlib.pyplot as plt
from src.utils import load_data
from src.constants import RF_FILE, GB_FILE, SEED, FIGURES_DIR
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def build_pipeline(model):
    """Wrap a model with StandardScaler so scaling is fit inside each CV fold (no test data leaks into training). Use this if you need the fitted estimator yourself, for example to draw ROC curves with CV."""
    return make_pipeline(StandardScaler(), model)

X, y = load_data()

best_rf = joblib.load(RF_FILE)
best_gb = joblib.load(GB_FILE)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEED
)

rf_pipe = build_pipeline(best_rf)
gb_pipe = build_pipeline(best_gb)

rf_pipe.fit(X_train, y_train)
gb_pipe.fit(X_train, y_train)

# Create one figure and one shared set of axes
fig, ax = plt.subplots(figsize=(8, 7))

RocCurveDisplay.from_estimator(
    rf_pipe,
    X_test,
    y_test,
    name="Random Forest",
    ax=ax
)

RocCurveDisplay.from_estimator(
    gb_pipe,
    X_test,
    y_test,
    name="Gradient Boosting",
    ax=ax
)

ax.set_title("ROC Curve Comparison: Random Forest vs. Gradient Boosting")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.grid(alpha=0.3)
ax.legend(loc="lower right")

fig.tight_layout()
plt.show()