import pandas as pd
from constants import DATA_PATH

df = pd.read_csv(DATA_PATH)

print(df.describe())
print("\nMissing values: ", df.isna().sum().sum())
print("Duplicate rows: ", df.duplicated().sum())
print("Class balance:\n", df["label"].value_counts())