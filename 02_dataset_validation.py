from pathlib import Path
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_PATH = SCRIPT_DIR.parent

CSV_PATH = PROJECT_PATH / "results" / "dataset_master.csv"


print("=" * 60)
print("DATASET VALIDATION")
print("=" * 60)

print("\nCSV path:")
print(CSV_PATH)

print("\nCSV exists:", CSV_PATH.exists())


if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"CSV not found:\n{CSV_PATH}"
    )


df = pd.read_csv(CSV_PATH)

print("\nTotal records:", len(df))



print("\nMissing values:")
print("-" * 60)

print(df.isnull().sum())



print("\nMine label distribution:")
print("-" * 60)

print(df["mine_label"].value_counts())



unknown = df[df["mine_label"] == -1]

print("\nUnknown-label records:", len(unknown))

if len(unknown) > 0:

    print("\nUnknown zones:")
    print(
        unknown["zone"]
        .value_counts()
    )



print("\nImages by date:")
print("-" * 60)

print(
    df["date"]
    .value_counts()
    .sort_index()
)


print("\nImages by format:")
print("-" * 60)

print(
    df["format"]
    .value_counts()
)


print("\nImages by zone:")
print("-" * 60)

print(
    df["zone"]
    .value_counts()
)


print("\nAltitude:")
print("-" * 60)

print(
    df["altitude_m"].describe()
)


print("\nTemperature:")
print("-" * 60)

print(
    df["temperature_c"].describe()
)

print("\n" + "=" * 60)
print("VALIDATION COMPLETED")
print("=" * 60)