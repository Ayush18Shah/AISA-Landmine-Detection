from pathlib import Path
import pandas as pd

# Locate dataset
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

CSV_PATH = PROJECT_DIR / "results" / "dataset_labeled.csv"
OUTPUT_PATH = PROJECT_DIR / "results" / "dataset_split.csv"

# Load labeled dataset
df = pd.read_csv(CSV_PATH)

# Keep only verified binary-label candidates.
# Labels are provisional and must be checked against the dataset documentation.
df = df[df["mine_label"].isin([0, 1])].copy()

# Assign groups by flight date
train_dates = ["13_02_2020", "17_02_2020", "20_02_2020"]
validation_dates = ["24_02_2020"]
test_dates = ["03_03_2020"]

def assign_split(date):
    if date in train_dates:
        return "train"
    elif date in validation_dates:
        return "validation"
    elif date in test_dates:
        return "test"
    else:
        return "unassigned"

df["split"] = df["date"].apply(assign_split)

# Display split sizes
print("\n--- Images per Split ---")
print(df["split"].value_counts())

print("\n--- Labels per Split ---")
print(pd.crosstab(df["split"], df["mine_label"]))

# Verify no date appears in multiple splits
print("\n--- Dates per Split ---")
print(pd.crosstab(df["date"], df["split"]))

# Save assignments
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("\nSplit file saved to:")
print(OUTPUT_PATH)

print("\nSplit preparation complete!")