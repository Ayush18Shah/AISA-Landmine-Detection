from pathlib import Path
import pandas as pd

# Locate the labeled dataset
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR.parent / "results" / "dataset_labeled.csv"

df = pd.read_csv(CSV_PATH)

print("TOTAL IMAGES:", len(df))

# Check label distribution
print("\n--- Label Distribution ---")
print(df["mine_label"].value_counts().sort_index())

# Check images by date and label
print("\n--- Date vs Label ---")
print(pd.crosstab(df["date"], df["mine_label"]))

# Check images by zone and label
print("\n--- Zone vs Label ---")
print(pd.crosstab(df["zone_clean"], df["mine_label"]))

# Check available formats by label
print("\n--- Format vs Label ---")
print(pd.crosstab(df["format"], df["mine_label"]))

print("\nSplit planning analysis complete!")