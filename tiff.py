from pathlib import Path
from PIL import Image
import numpy as np

# Find the dataset directory
BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = BASE_DIR / "DatainBrief_dataset_1"

if not DATASET_DIR.exists():
    DATASET_DIR = BASE_DIR

# Find TIFF files, including TIFF images with unusual extensions
image_files = [
    p for p in DATASET_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower() in {".tif", ".tiff"}
]

# Some TIFF images may have a .jpg extension
for p in DATASET_DIR.rglob("*.jpg"):
    try:
        with Image.open(p) as image:
            if image.format == "TIFF":
                image_files.append(p)
    except Exception:
        pass

print(f"Total TIFF images found: {len(image_files)}")

for path in image_files[:10]:
    try:
        with Image.open(path) as image:
            array = np.array(image)

            print("\nFile:", path.name)
            print("Dimensions:", image.size)
            print("Image mode:", image.mode)
            print("Data type:", array.dtype)
            print("Pixel range:", array.min(), "to", array.max())
            print("Array shape:", array.shape)

    except Exception as error:
        print(f"Could not read {path.name}: {error}")