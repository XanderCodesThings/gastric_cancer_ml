import matplotlib.pyplot as plt
from src.utils import compare
from src.random_forest import best_rf
from src.gradient_boosting import best_gbdt
from src.constants import FIGURES_DIR

results = compare({
    "Random Forest": best_rf,
    "Gradient Boosting": best_gbdt
})

ax = results.plot(
    kind="bar",
    figsize=(10,6),
    rot=0
)

plt.title("Comparison of Tuned Model Performance")
plt.ylabel("Score")
plt.ylim(0.7, 1.0)
plt.grid(axis="y", alpha=0.3)
plt.legend(title="Metric", bbox_to_anchor=(1.02,1))
plt.tight_layout()

plt.savefig(FIGURES_DIR / "metrics-barchart.png")