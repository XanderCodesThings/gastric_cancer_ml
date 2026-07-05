import matplotlib.pyplot as plt
import pandas as pd
from src.constants import FIGURES_DIR, METRICS_FILE

results = pd.read_csv(METRICS_FILE, index_col=0)
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

# plt.show()

plt.savefig(FIGURES_DIR / "metrics-barchart.png")