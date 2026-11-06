from pathlib import Path
import csv
import numpy as np
from PIL import Image

# Project directory
BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = BASE_DIR / "DatainBrief_dataset_1"

if not DATASET_DIR.exists():
    DATASET_DIR = BASE_DIR

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

output_csv = RESULTS_DIR / "thermal_image_index.csv"

records = []

for path in DATASET_DIR.rglob("*"):
    if not path.is_file():
        continue

    try:
        with Image.open(path) as img:
            # Detect TIFF files even when they have a .jpg extension
            if img.format != "TIFF":
                continue

            array = np.array(img)

            records.append({
                "filename": path.name,
                "filepath": str(path.resolve()),
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "dtype": str(array.dtype),
                "min_pixel": int(array.min()),
                "max_pixel": int(array.max()),
                "label": ""
            })

    except Exception:
        continue

with open(output_csv, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=[
        "filename",
        "filepath",
        "width",
        "height",
        "mode",
        "dtype",
        "min_pixel",
        "max_pixel",
        "label"
    ])

    writer.writeheader()
    writer.writerows(records)

print("Thermal image inspection completed.")
print("Images indexed:", len(records))
print("Index saved to:", output_csv)
print("Original images have not been modified.")
print("Labels must be verified before supervised training.")