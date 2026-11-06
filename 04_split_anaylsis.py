from pathlib import Path
import pandas as pd

# Locate the dataset CSV
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR.parent / "results" / "dataset_master.csv"

# Load dataset
df = pd.read_csv(CSV_PATH)

# Standardize zone names
def clean_zone(zone):
    if pd.isna(zone):
        return "Unknown"

    zone = str(zone).lower()

    for number in range(1, 10):
        if f"zone {number}" in zone:
            return f"Zone {number}"

    return "Unknown"


df["zone_clean"] = df["zone"].apply(clean_zone)

# Analyze images by date and zone
print("\n--- Images by Date and Zone ---")

summary = pd.crosstab(
    df["date"],
    df["zone_clean"]
)

print(summary)

# Analyze images by date and format
print("\n--- Images by Date and Format ---")

format_summary = pd.crosstab(
    df["date"],
    df["format"]
)

print(format_summary)

print("\nTotal images:", len(df))
print("\nAnalysis complete!")