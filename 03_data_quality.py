from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR.parent / "results" / "dataset_master.csv"

df = pd.read_csv(CSV_PATH)

# Create clean zone labels
def clean_zone(zone):
    if pd.isna(zone):
        return "Unknown"

    zone = str(zone).lower()

    if "zone 1" in zone:
        return "Zone 1"
    elif "zone 2" in zone:
        return "Zone 2"
    elif "zone 3" in zone:
        return "Zone 3"
    elif "zone 4" in zone:
        return "Zone 4"
    elif "zone 5" in zone:
        return "Zone 5"
    elif "zone 6" in zone:
        return "Zone 6"
    elif "zone 7" in zone:
        return "Zone 7"
    elif "zone 8" in zone:
        return "Zone 8"
    elif "zone 9" in zone:
        return "Zone 9"
    else:
        return "Unknown"


df["zone_clean"] = df["zone"].apply(clean_zone)

print("Original zone count:")
print(df["zone"].value_counts())

print("\nClean zone count:")
print(df["zone_clean"].value_counts().sort_index())

print("\nDONE!")