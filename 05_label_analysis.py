from pathlib import Path
import pandas as pd

# Locate dataset
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

CSV_PATH = PROJECT_DIR / "results" / "dataset_master.csv"
OUTPUT_PATH = PROJECT_DIR / "results" / "dataset_labeled.csv"

# Load dataset
df = pd.read_csv(CSV_PATH)

# Clean zone names
def clean_zone(zone):
    if pd.isna(zone):
        return "Unknown"

    zone = str(zone).lower()

    for number in range(1, 10):
        if f"zone {number}" in zone:
            return f"Zone {number}"

    return "Unknown"


df["zone_clean"] = df["zone"].apply(clean_zone)

# Assign provisional labels
def assign_label(zone):
    if zone == "Zone 6":
        return 0
    elif zone.startswith("Zone "):
        return 1
    else:
        return -1


df["mine_label"] = df["zone_clean"].apply(assign_label)

# Display results
print("\n--- Label Distribution ---")
print(df["mine_label"].value_counts().sort_index())

print("\n--- Clean Zone Distribution ---")
print(df["zone_clean"].value_counts().sort_index())

# Save labeled dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\nTotal images:", len(df))
print("Unknown-label images:", (df["mine_label"] == -1).sum())
print("\nLabeled dataset saved to:")
print(OUTPUT_PATH)

print("\nAnalysis complete!")