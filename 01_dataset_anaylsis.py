from pathlib import Path
import re
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

DATASET_PATH = SCRIPT_DIR.parent / "DatainBrief_dataset_1"

print("Dataset path:")
print(DATASET_PATH)

print("\nDataset exists:", DATASET_PATH.exists())


if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at:\n{DATASET_PATH}"
    )


extensions = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}

image_files = [
    file
    for file in DATASET_PATH.rglob("*")
    if file.is_file()
    and file.suffix.lower() in extensions
]

print("\nTotal images:", len(image_files))


records = []

for file in image_files:

    parts = file.relative_to(DATASET_PATH).parts

    date = parts[0] if len(parts) > 0 else None
    format_folder = parts[1] if len(parts) > 1 else None
    zone = parts[2] if len(parts) > 2 else None

    
    altitude_match = re.search(
        r"FlightAltitude__([-+]?\d+(?:\.\d+)?)m",
        file.name,
        re.IGNORECASE
    )

    altitude = (
        float(altitude_match.group(1))
        if altitude_match
        else None
    )

    
    temperature_match = re.search(
        r"TemperatureInCenter__([-+]?\d+(?:\.\d+)?)°C",
        file.name,
        re.IGNORECASE
    )

    temperature = (
        float(temperature_match.group(1))
        if temperature_match
        else None
    )


    if zone and "free zone" in zone.lower():
        mine_label = 0
    elif zone and "mine" in zone.lower():
        mine_label = 1
    else:
        mine_label = -1

    records.append({
        "filepath": str(file),
        "filename": file.name,
        "date": date,
        "format": format_folder,
        "zone": zone,
        "altitude_m": altitude,
        "temperature_c": temperature,
        "mine_label": mine_label
    })


df = pd.DataFrame(records)

print("\nDataset table created!")
print("Rows:", len(df))

print("\nFirst 5 rows:")
print(df.head())


results_folder = SCRIPT_DIR.parent / "results"
results_folder.mkdir(exist_ok=True)

output_file = results_folder / "dataset_master.csv"

df.to_csv(output_file, index=False)

print("\nCSV saved to:")
print(output_file)

print("\nDONE!")