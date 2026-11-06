from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

CSV_PATH = PROJECT_DIR / "results" / "dataset_split.csv"
OUTPUT_DIR = PROJECT_DIR / "processed" / "rgb"
OUTPUT_CSV = PROJECT_DIR / "results" / "rgb_preprocessed.csv"

IMAGE_SIZE = (224, 224)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)

# Keep JPG and R_JPG images with known binary labels
df = df[
    df["format"].isin(["JPG", "R_JPG"])
    & df["mine_label"].isin([0, 1])
].copy()

print("RGB images to process:", len(df))

# --------------------------------------------------
# 3. PREPROCESS IMAGES
# --------------------------------------------------

records = []
errors = []

for index, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Preprocessing RGB images"
):
    source_path = Path(row["filepath"])

    # Keep each image in its assigned split folder
    split = row["split"]
    output_folder = OUTPUT_DIR / split
    output_folder.mkdir(parents=True, exist_ok=True)

    # Use a unique output name based on the source path
    output_name = f"{index:05d}_{source_path.stem}.png"
    output_path = output_folder / output_name

    try:
        with Image.open(source_path) as img:
            # Standardize image representation
            img = img.convert("RGB")

            # Resize all images to a consistent size
            img = img.resize(
                IMAGE_SIZE,
                Image.Resampling.LANCZOS
            )

            # Save processed image without changing the original
            img.save(output_path, format="PNG")

        records.append({
            "source_filepath": str(source_path),
            "processed_filepath": str(output_path),
            "date": row["date"],
            "format": row["format"],
            "zone_clean": row["zone_clean"],
            "mine_label": int(row["mine_label"]),
            "split": split,
            "width": IMAGE_SIZE[0],
            "height": IMAGE_SIZE[1]
        })

    except Exception as error:
        errors.append({
            "filepath": str(source_path),
            "error": str(error)
        })

# --------------------------------------------------
# 4. SAVE PROCESSING INDEX
# --------------------------------------------------

processed_df = pd.DataFrame(records)
processed_df.to_csv(OUTPUT_CSV, index=False)

# --------------------------------------------------
# 5. REPORT RESULTS
# --------------------------------------------------

print("\n--- PREPROCESSING SUMMARY ---")
print("Images selected:", len(df))
print("Successfully processed:", len(processed_df))
print("Failed images:", len(errors))

if not processed_df.empty:
    print("\nImages by split:")
    print(processed_df["split"].value_counts())

    print("\nLabels by split:")
    print(pd.crosstab(
        processed_df["split"],
        processed_df["mine_label"]
    ))

if errors:
    error_path = PROJECT_DIR / "results" / "rgb_preprocessing_errors.csv"
    pd.DataFrame(errors).to_csv(error_path, index=False)
    print("\nError report saved to:", error_path)

print("\nProcessed image index saved to:")
print(OUTPUT_CSV)

print("\nRGB preprocessing complete!")