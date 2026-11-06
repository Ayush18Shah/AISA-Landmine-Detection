from pathlib import Path
import pandas as pd
from PIL import Image

# -----------------------------------------
# 1. LOCATE FILES
# -----------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

CSV_PATH = PROJECT_DIR / "results" / "rgb_preprocessed.csv"

# -----------------------------------------
# 2. LOAD PROCESSED DATASET
# -----------------------------------------

df = pd.read_csv(CSV_PATH)

print("Total processed images:", len(df))

# -----------------------------------------
# 3. CHECK FILES AND IMAGE PROPERTIES
# -----------------------------------------

missing_files = []
wrong_dimensions = []
wrong_mode = []
unreadable_files = []

for filepath in df["processed_filepath"]:
    path = Path(filepath)

    if not path.exists():
        missing_files.append(str(path))
        continue

    try:
        with Image.open(path) as img:
            img.verify()

        # Reopen after verify to inspect image properties
        with Image.open(path) as img:
            if img.size != (224, 224):
                wrong_dimensions.append(str(path))

            if img.mode != "RGB":
                wrong_mode.append(str(path))

    except Exception:
        unreadable_files.append(str(path))

# -----------------------------------------
# 4. CHECK LABELS AND SPLITS
# -----------------------------------------

print("\n--- Label Distribution ---")
print(df["mine_label"].value_counts().sort_index())

print("\n--- Images per Split ---")
print(df["split"].value_counts())

print("\n--- Labels per Split ---")
print(pd.crosstab(df["split"], df["mine_label"]))

# -----------------------------------------
# 5. DISPLAY VERIFICATION RESULTS
# -----------------------------------------

print("\n--- Verification Results ---")
print("Missing files:", len(missing_files))
print("Unreadable files:", len(unreadable_files))
print("Incorrect dimensions:", len(wrong_dimensions))
print("Incorrect image mode:", len(wrong_mode))

if not any([
    missing_files,
    unreadable_files,
    wrong_dimensions,
    wrong_mode
]):
    print("\nSUCCESS: All processed images passed verification!")
else:
    print("\nWARNING: Some images need further inspection.")

print("\nVerification complete!")
