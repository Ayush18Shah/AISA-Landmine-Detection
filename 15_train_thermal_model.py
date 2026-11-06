from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

CSV_PATH = BASE_DIR / "results" / "thermal_image_index.csv"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "thermal_cnn_model.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

# --------------------------------------------------
# 2. Load and validate the dataset index
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)

required_columns = {"filepath", "date", "label"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"CSV must contain these columns: {required_columns}"
    )

df["label"] = df["label"].astype(int)

if not set(df["label"].unique()).issubset({0, 1}):
    raise ValueError("Unexpected labels. Expected 0 and 1.")

# Use only files that still exist
df = df[df["filepath"].apply(lambda p: Path(p).is_file())].copy()

print("\nImages available:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts().sort_index())

# --------------------------------------------------
# 3. Split by collection date
# --------------------------------------------------

train_dates = ["13_02_2020", "17_02_2020"]
validation_dates = ["24_02_2020"]
test_dates = ["03_03_2020"]

train_df = df[df["date"].isin(train_dates)].copy()
val_df = df[df["date"].isin(validation_dates)].copy()
test_df = df[df["date"].isin(test_dates)].copy()

if min(len(train_df), len(val_df), len(test_df)) == 0:
    raise ValueError(
        "One split is empty. Check the date values in the CSV."
    )

# Every split must contain both classes for meaningful binary evaluation
for split_name, split_df in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Test", test_df),
]:
    print(f"\n{split_name} distribution:")
    print(split_df["label"].value_counts().sort_index())

    if set(split_df["label"].unique()) != {0, 1}:
        raise ValueError(
            f"{split_name} split does not contain both classes. "
            "Review the date split before training."
        )

# Save the split information
split_records = []

for split_name, split_df in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df),
]:
    for _, row in split_df.iterrows():
        split_records.append({
            "filepath": row["filepath"],
            "date": row["date"],
            "label": row["label"],
            "split": split_name,
        })

pd.DataFrame(split_records).to_csv(
    RESULTS_DIR / "thermal_dataset_split.csv",
    index=False,
)

# --------------------------------------------------
# 4. Thermal image dataset
# --------------------------------------------------

class ThermalDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        with Image.open(row["filepath"]) as image:
            array = np.array(image)

        if array.ndim != 2:
            raise ValueError(
                f"Expected a single-channel image: {row['filepath']}"
            )

        # Preserve 16-bit information while normalizing each image
        array = array.astype(np.float32)

        low = float(array.min())
        high = float(array.max())

        if high > low:
            array = (array - low) / (high - low)
        else:
            array = np.zeros_like(array, dtype=np.float32)

        image = Image.fromarray(array, mode="F")
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR,
        )

        array = np.array(image, dtype=np.float32)

        # Add one channel: [1, height, width]
        tensor = torch.from_numpy(array).unsqueeze(0)

        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long,
        )

        return tensor, label


train_dataset = ThermalDataset(train_df)
val_dataset = ThermalDataset(val_df)
test_dataset = ThermalDataset(test_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# --------------------------------------------------
# 5. Define a thermal CNN
# --------------------------------------------------

class ThermalCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = ThermalCNN().to(DEVICE)

# Account for class imbalance in the training data
class_counts = train_df["label"].value_counts()

class_weights = torch.tensor(
    [
        len(train_df) / (2 * class_counts[0]),
        len(train_df) / (2 * class_counts[1]),
    ],
    dtype=torch.float32,
    device=DEVICE,
)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

# --------------------------------------------------
# 6. Evaluation function
# --------------------------------------------------

def evaluate(loader):
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().numpy()

            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.numpy().tolist())

    accuracy = accuracy_score(all_labels, all_predictions)

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return accuracy, macro_f1, all_labels, all_predictions


# --------------------------------------------------
# 7. Train the model
# --------------------------------------------------

best_val_f1 = -1.0

print("\nStarting thermal CNN training...\n")

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    train_loss = total_loss / len(train_dataset)

    val_accuracy, val_f1, _, _ = evaluate(val_loader)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Validation Accuracy: {val_accuracy:.4f} | "
        f"Validation Macro F1: {val_f1:.4f}"
    )

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "image_size": IMAGE_SIZE,
                "class_names": {
                    0: "Free-zone dataset label",
                    1: "Other mine-related dataset label",
                },
                "normalization": "Per-image min-max normalization",
                "input_channels": 1,
                "train_dates": train_dates,
                "validation_dates": validation_dates,
                "test_dates": test_dates,
            },
            MODEL_PATH,
        )

# --------------------------------------------------
# 8. Load best validation checkpoint and test once
# --------------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])

test_accuracy, test_f1, y_true, y_pred = evaluate(test_loader)

print("\nFINAL TEST RESULTS")
print("------------------")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Macro F1: {test_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Free-zone label", "Other zone label"],
        zero_division=0,
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nThermal model saved to:")
print(MODEL_PATH)

print("\nDate split saved to:")
print(RESULTS_DIR / "thermal_dataset_split.csv")