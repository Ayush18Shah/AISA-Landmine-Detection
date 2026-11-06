from pathlib import Path
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

CSV_PATH = PROJECT_DIR / "results" / "rgb_preprocessed.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "rgb_cnn_model.pth"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
IMAGE_SIZE = 224

torch.manual_seed(42)
np.random.seed(42)

print("Training device:", DEVICE)


# ==========================================
# 2. LOAD DATASET INDEX
# ==========================================

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find {CSV_PATH}\n"
        "Run 11_rgb_preprocessing.py first."
    )

df = pd.read_csv(CSV_PATH)

required_columns = {
    "processed_filepath",
    "mine_label",
    "split",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing CSV columns: {missing_columns}")

# Exclude any rows without a valid binary label.
df = df[df["mine_label"].isin([0, 1])].copy()

print("\nDataset size:", len(df))
print("\nLabel distribution:")
print(df["mine_label"].value_counts().sort_index())


# ==========================================
# 3. IMAGE TRANSFORMATION
# ==========================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


# ==========================================
# 4. DATASET CLASS
# ==========================================

class RGBImageDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

        # Check paths before training.
        missing = [
            p for p in self.dataframe["processed_filepath"]
            if not Path(p).is_file()
        ]

        if missing:
            raise FileNotFoundError(
                f"{len(missing)} processed image files are missing. "
                f"Example: {missing[0]}"
            )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = Path(row["processed_filepath"])
        label = int(row["mine_label"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ==========================================
# 5. PREPARE TRAINING, VALIDATION AND TEST SETS
# ==========================================

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

if train_df.empty or val_df.empty or test_df.empty:
    raise ValueError(
        "One or more splits are empty. "
        "Check the 'split' column in rgb_preprocessed.csv."
    )

print("\nTraining images:", len(train_df))
print("Validation images:", len(val_df))
print("Test images:", len(test_df))

train_dataset = RGBImageDataset(train_df, transform)
val_dataset = RGBImageDataset(val_df, transform)
test_dataset = RGBImageDataset(test_df, transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ==========================================
# 6. DEFINE A SMALL CNN
# ==========================================

class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
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
        x = self.features(x)
        return self.classifier(x)


model = SimpleCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ==========================================
# 7. TRAINING AND VALIDATION
# ==========================================

best_val_loss = float("inf")
best_model_weights = copy.deepcopy(model.state_dict())

print("\nStarting model training...\n")

for epoch in range(EPOCHS):

    # ---- Training ----
    model.train()

    train_loss_total = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss_total += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    train_loss = train_loss_total / train_total
    train_accuracy = train_correct / train_total

    # ---- Validation ----
    model.eval()

    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_total += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss_total / val_total
    val_accuracy = val_correct / val_total

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f}"
    )

    # Save the weights with the lowest validation loss.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = copy.deepcopy(model.state_dict())


# ==========================================
# 8. SAVE BEST MODEL
# ==========================================

model.load_state_dict(best_model_weights)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "image_size": IMAGE_SIZE,
        "class_names": {
            0: "free_zone",
            1: "mine_related",
        },
        "normalization_mean": [0.5, 0.5, 0.5],
        "normalization_std": [0.5, 0.5, 0.5],
    },
    MODEL_PATH,
)

print("\nBest model saved to:")
print(MODEL_PATH)


# ==========================================
# 9. EVALUATE ON HELD-OUT TEST SET
# ==========================================

model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy().tolist())
        all_predictions.extend(predictions.cpu().numpy().tolist())

accuracy = accuracy_score(all_labels, all_predictions)
f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0,
)

print("\n" + "=" * 50)
print("FINAL TEST RESULTS")
print("=" * 50)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Macro F1-score: {f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1],
        target_names=["Free zone", "Mine-related"],
        zero_division=0,
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1],
    )
)

print("\nTraining and evaluation complete!")