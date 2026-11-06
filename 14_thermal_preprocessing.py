from pathlib import Path
import csv

import numpy as np
from PIL import Image

# Project root
BASE_DIR = Path(__file__).resolve().parents[1]

# Dataset location
DATASET_DIR = BASE_DIR / "DatainBrief_dataset_1"

if not DATASET_DIR.exists():
    DATASET_DIR = BASE_DIR

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = RESULTS_DIR / "thermal_image_index.csv"

records = []
skipped = 0

for path in DATASET_DIR.rglob("*"):
    if not path.is_file():
        continue

    try:
        with Image.open(path) as image:
            # Some TIFF images have a .jpg extension
            if image.format != "TIFF":
                continue

            array = np.array(image)

            # Accept single-channel numeric thermal-like images
            if array.ndim != 2:
                skipped += 1
                continue

            relative_parts = path.relative_to(DATASET_DIR).parts

            # Find the zone folder in the path
            zone_folder = next(
                (
                    part for part in relative_parts
                    if part.lower().startswith("zone ")
                ),
                None
            )

            if zone_folder is None:
                skipped += 1
                continue

            # Assign provisional dataset-derived labels
            if "zone 6" in zone_folder.lower():
                label = 0
            else:
                label = 1

            date_folder = relative_parts[0]

            records.append({
                "filename": path.name,
                "filepath": str(path.resolve()),
                "date": date_folder,
                "zone": zone_folder,
                "width": image.width,
                "height": image.height,
                "dtype": str(array.dtype),
                "min_pixel": int(array.min()),
                "max_pixel": int(array.max()),
                "label": label
            })

    except Exception:
        skipped += 1

fieldnames = [
    "filename",
    "filepath",
    "date",
    "zone",
    "width",
    "height",
    "dtype",
    "min_pixel",
    "max_pixel",
    "label"
]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

print("\nTHERMAL DATASET INDEX CREATED")
print("Images indexed:", len(records))
print("Images skipped:", skipped)
print("Output:", OUTPUT_CSV)

print("\nCLASS COUNTS")

for label in [0, 1]:
    count = sum(row["label"] == label for row in records)
    print(f"Class {label}: {count} images")

print("\nDATE COUNTS")

for date in sorted(set(row["date"] for row in records)):
    count = sum(row["date"] == date for row in records)
    print(f"{date}: {count} images")