from pathlib import Path
import pandas as pd

# Locate the labeled dataset
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR.parent / "results" / "dataset_labeled.csv"

df = pd.read_csv(CSV_PATH)

print("Total images:", len(df))

# Inspect available information
print("\n--- Available Columns ---")
print(df.columns.tolist())

# Show example image paths
print("\n--- Sample Image Paths ---")
for path in df["filepath"].head(10):
    print(path)

# Count images by date and zone
print("\n--- Images by Date and Clean Zone ---")
print(
    pd.crosstab(
        df["date"],
        df["zone_clean"]
    )
)

print("\nLeakage investigation: initial inspection complete!")