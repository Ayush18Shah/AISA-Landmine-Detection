from pathlib import Path
from PIL import Image
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR.parent / "results" / "dataset_labeled.csv"

df = pd.read_csv(CSV_PATH)


# Calculate a difference hash (dHash)
def image_hash(filepath):
    try:
        with Image.open(filepath) as img:
            img = img.convert("L").resize((9, 8))
            pixels = list(img.getdata())

        bits = []

        for row in range(8):
            for col in range(8):
                left = pixels[row * 9 + col]
                right = pixels[row * 9 + col + 1]
                bits.append("1" if left > right else "0")

        return "".join(bits)

    except Exception:
        return None


print("Checking images using dHash...")
print("Total images:", len(df))

df["visual_hash"] = df["filepath"].apply(image_hash)

valid = df.dropna(subset=["visual_hash"])

duplicates = valid[
    valid.duplicated("visual_hash", keep=False)
].sort_values("visual_hash")

print("\nImages successfully checked:", len(valid))
print("Images that could not be checked:", len(df) - len(valid))
print("Potential matching images:", len(duplicates))

if not duplicates.empty:
    print("\nPotential matches (first 20):")

    print(
        duplicates[
            ["filename", "date", "zone", "visual_hash"]
        ].head(20).to_string(index=False)
    )
else:
    print("No matching hashes found.")

print("\nCheck complete!")