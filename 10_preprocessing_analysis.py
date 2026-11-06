from pathlib import Path
from PIL import Image
import pandas as pd

# Locate the dataset
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR.parent / "results" / "dataset_split.csv"

df = pd.read_csv(CSV_PATH)

# Store image information
records = []

print("Inspecting all images...")
print("Total images:", len(df))

for _, row in df.iterrows():
    filepath = row["filepath"]

    try:
        with Image.open(filepath) as img:
            records.append({
                "filename": Path(filepath).name,
                "folder_format": row["format"],
                "actual_format": img.format,
                "width": img.width,
                "height": img.height,
                "mode": img.mode
            })

    except Exception as error:
        records.append({
            "filename": Path(filepath).name,
            "folder_format": row["format"],
            "actual_format": "ERROR",
            "width": None,
            "height": None,
            "mode": str(error)
        })

# Create a summary table
results = pd.DataFrame(records)

print("\n--- Actual Image Formats ---")
print(results["actual_format"].value_counts())

print("\n--- Image Modes ---")
print(results["mode"].value_counts())

print("\n--- Image Dimensions ---")
print(
    results.groupby(["width", "height"], dropna=False)
    .size()
    .sort_values(ascending=False)
)

print("\n--- Folder Format vs Actual Format ---")
print(pd.crosstab(
    results["folder_format"],
    results["actual_format"]
))

# Save inspection results
OUTPUT_PATH = SCRIPT_DIR.parent / "results" / "image_inspection.csv"
results.to_csv(OUTPUT_PATH, index=False)

print("\nImages with inspection errors:",
      (results["actual_format"] == "ERROR").sum())

print("\nInspection report saved to:")
print(OUTPUT_PATH)

print("\nInspection complete!")

# Summarize the existing inspection report

from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_PATH = SCRIPT_DIR.parent / "results" / "image_inspection.csv"

results = pd.read_csv(REPORT_PATH)

print("\n--- Actual Image Formats ---")
print(results["actual_format"].value_counts())

print("\n--- Image Modes ---")
print(results["mode"].value_counts())

print("\n--- Image Dimensions ---")
print(
    results.groupby(["width", "height"], dropna=False)
    .size()
    .sort_values(ascending=False)
)

print("\n--- Folder Format vs Actual Format ---")
print(pd.crosstab(results["folder_format"], results["actual_format"]))

print("\nInspection errors:",
      (results["actual_format"] == "ERROR").sum())