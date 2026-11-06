from flask import Flask, render_template, request, jsonify
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps
import uuid
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# =====================================================
# 1. FLASK CONFIGURATION
# =====================================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = BASE_DIR.parent / "DatainBrief_dataset_1" / "models"
RESULTS_DIR = MODEL_DIR.parent / "results"

RGB_MODEL_PATH = MODEL_DIR / "rgb_cnn_model.pth"
THERMAL_MODEL_PATH = MODEL_DIR / "thermal_cnn_model.pth"

RGB_CSV_PATH = RESULTS_DIR / "rgb_preprocessed.csv"

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)


# =====================================================
# 2. CNN ARCHITECTURE
# =====================================================

class SimpleCNN(nn.Module):

    def __init__(self, input_channels=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
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


# =====================================================
# 3. IMAGE TRANSFORMS
# =====================================================

rgb_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


def preprocess_rgb(image):
    image = image.convert("RGB")
    return rgb_transform(image).unsqueeze(0)


def preprocess_thermal(image):

    image_array = np.asarray(image)

    if image_array.ndim != 2:
        raise ValueError(
            "Thermal model requires a single-channel thermal image."
        )

    image_array = image_array.astype(np.float32)

    minimum = float(image_array.min())
    maximum = float(image_array.max())

    if maximum > minimum:
        image_array = (
            (image_array - minimum) / (maximum - minimum)
        )
    else:
        image_array = np.zeros_like(
            image_array,
            dtype=np.float32,
        )

    thermal_image = Image.fromarray(image_array, mode="F")

    thermal_image = thermal_image.resize(
        (224, 224),
        Image.Resampling.BILINEAR,
    )

    tensor = transforms.ToTensor()(thermal_image)

    return tensor.unsqueeze(0)


# =====================================================
# 4. LOAD TRAINED MODELS
# =====================================================

def load_model(model_path, input_channels):

    if not model_path.exists():
        logging.error("Model file not found: %s", model_path)
        return None

    try:
        model = SimpleCNN(input_channels=input_channels)

        try:
            checkpoint = torch.load(
                model_path,
                map_location=DEVICE,
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(
                model_path,
                map_location=DEVICE,
            )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        logging.info("Loaded model: %s", model_path.name)

        return model

    except Exception:
        logging.exception("Failed to load model: %s", model_path)
        return None


rgb_model = load_model(
    RGB_MODEL_PATH,
    input_channels=3,
)

thermal_model = load_model(
    THERMAL_MODEL_PATH,
    input_channels=1,
)


print("\n========== AISA MODEL STATUS ==========")
print("RGB model connected:", rgb_model is not None)
print("Thermal model connected:", thermal_model is not None)
print("Device:", DEVICE)
print("RGB evaluation CSV exists:", RGB_CSV_PATH.exists())
print("=======================================\n")


# =====================================================
# 5. PREDICTION
# =====================================================

CLASS_LABELS = {
    0: "Free zone",
    1: "Mine-related",
}


def predict(model, tensor):

    tensor = tensor.to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)

        predicted_class = int(
            torch.argmax(probabilities, dim=1).item()
        )

        score = float(
            probabilities[0, predicted_class].item() * 100
        )

    return predicted_class, score


# =====================================================
# 6. SAVE IMAGE PREVIEWS
# =====================================================

def save_preview(image, filename):

    path = UPLOAD_DIR / filename

    image.save(
        path,
        format="JPEG",
        quality=90,
    )

    return f"/static/uploads/{filename}"


# =====================================================
# 7. MODEL PERFORMANCE EVALUATION
# =====================================================

performance_cache = None


def resolve_dataset_path(value):

    path = Path(str(value))

    if path.is_absolute():
        return path

    # First try the path as stored in the CSV.
    if path.is_file():
        return path

    # Then try relative to the dataset directory.
    candidate = MODEL_DIR.parent / path

    if candidate.is_file():
        return candidate

    # Finally try relative to the project directory.
    candidate = BASE_DIR / path

    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"Test image path not found: {value}"
    )


def evaluate_rgb_model():

    global performance_cache

    if performance_cache is not None:
        return performance_cache

    if rgb_model is None:
        raise RuntimeError(
            "RGB model failed to load. Check the Flask terminal."
        )

    if not RGB_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation CSV not found: {RGB_CSV_PATH}"
        )

    df = pd.read_csv(RGB_CSV_PATH)

    required_columns = {
        "processed_filepath",
        "mine_label",
        "split",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing_columns)}"
        )

    df = df[
        df["mine_label"].isin([0, 1])
    ].copy()

    test_df = df[
        df["split"].astype(str).str.strip().str.lower() == "test"
    ].copy()

    if test_df.empty:
        raise ValueError(
            "No test samples were found in rgb_preprocessed.csv. "
            "Check the values in its split column."
        )

    y_true = []
    y_pred = []

    rgb_model.eval()

    for _, row in test_df.iterrows():

        image_path = resolve_dataset_path(
            row["processed_filepath"]
        )

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = preprocess_rgb(image)

        predicted_class, _ = predict(
            rgb_model,
            tensor,
        )

        y_true.append(int(row["mine_label"]))
        y_pred.append(predicted_class)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    performance_cache = {
        "accuracy": round(float(accuracy) * 100, 2),
        "precision": round(float(precision) * 100, 2),
        "recall": round(float(recall) * 100, 2),
        "f1_score": round(float(f1) * 100, 2),
        "test_samples": len(y_true),
        "confusion_matrix": matrix.tolist(),
        "positive_class": "Mine-related (label 1)",
        "evaluation_split": "test",
    }

    logging.info(
        "RGB test evaluation complete: %s",
        performance_cache,
    )

    return performance_cache


# =====================================================
# 8. HOME PAGE
# =====================================================

@app.route("/")
def home():
    return render_template("index.html")


# =====================================================
# 9. ANALYZE IMAGE
# =====================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "Please select an image.",
        }), 400

    file = request.files["image"]

    if not file.filename:
        return jsonify({
            "success": False,
            "error": "No image selected.",
        }), 400

    if "." not in file.filename:
        return jsonify({
            "success": False,
            "error": "Invalid filename.",
        }), 400

    extension = file.filename.rsplit(".", 1)[-1].lower()

    model_type = request.form.get(
        "model_type",
        "rgb",
    ).lower()

    if model_type not in {"rgb", "thermal"}:
        return jsonify({
            "success": False,
            "error": "Invalid model selection.",
        }), 400

    if model_type == "rgb" and extension not in {
        "jpg", "jpeg", "png"
    }:
        return jsonify({
            "success": False,
            "error": "RGB model accepts JPG, JPEG, and PNG images.",
        }), 400

    if model_type == "thermal" and extension not in {
        "tif", "tiff"
    }:
        return jsonify({
            "success": False,
            "error": "Thermal model requires a TIFF image.",
        }), 400

    model = rgb_model if model_type == "rgb" else thermal_model

    if model is None:
        return jsonify({
            "success": False,
            "model_connected": False,
            "error": f"{model_type.upper()} model failed to load.",
        }), 503

    try:
        file.stream.seek(0)

        with Image.open(file.stream) as opened_image:
            actual_format = opened_image.format
            opened_image.verify()

        file.stream.seek(0)

        with Image.open(file.stream) as opened_image:
            image = opened_image.copy()

        if model_type == "rgb":

            if actual_format not in {"JPEG", "PNG"}:
                return jsonify({
                    "success": False,
                    "error": "Please upload a valid JPG or PNG image.",
                }), 400

            input_tensor = preprocess_rgb(image)

        else:

            if actual_format != "TIFF":
                return jsonify({
                    "success": False,
                    "error": "Thermal image must contain actual TIFF data.",
                }), 400

            input_tensor = preprocess_thermal(image)

        predicted_class, score = predict(
            model,
            input_tensor,
        )

        # Generate previews for the preprocessing cards.
        original_preview = image.convert("RGB")
        grayscale_preview = ImageOps.grayscale(image)
        filtered_preview = grayscale_preview.filter(
            ImageFilter.MedianFilter(size=3)
        )

        preview_id = uuid.uuid4().hex

        original_url = save_preview(
            original_preview,
            f"{preview_id}_original.jpg",
        )

        grayscale_url = save_preview(
            grayscale_preview,
            f"{preview_id}_grayscale.jpg",
        )

        filtered_url = save_preview(
            filtered_preview,
            f"{preview_id}_filtered.jpg",
        )

        model_name = (
            "RGB CNN"
            if model_type == "rgb"
            else "Thermal CNN"
        )

        message = (
            f"{model_name} generated an experimental classification. "
            "This prediction does not confirm landmine presence or absence."
        )

        return jsonify({
            "success": True,
            "model_connected": True,
            "model_type": model_type,
            "model_name": model_name,

            "image_url": original_url,
            "original_url": original_url,
            "grayscale_url": grayscale_url,
            "filtered_url": filtered_url,

            "prediction": CLASS_LABELS[predicted_class],
            "predicted_class": predicted_class,
            "model_score": round(score, 2),

            "preprocessing_complete": True,
            "feature_analysis_complete": True,

            "message": message,
        })

    except ValueError as error:

        return jsonify({
            "success": False,
            "error": str(error),
        }), 400

    except Exception:

        logging.exception("Image analysis failed")

        return jsonify({
            "success": False,
            "error": "Image analysis failed. Check the Flask terminal.",
        }), 500


# =====================================================
# 10. MODEL PERFORMANCE API
# =====================================================

@app.route("/model-performance", methods=["GET"])
def model_performance():

    try:
        metrics = evaluate_rgb_model()

        return jsonify({
            "success": True,
            **metrics,
        })

    except Exception as error:

        logging.exception(
            "Model performance evaluation failed"
        )

        return jsonify({
            "success": False,
            "error": str(error),
        }), 500


# =====================================================
# 11. UPLOAD SIZE ERROR
# =====================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "success": False,
        "error": "Image must be smaller than 10 MB.",
    }), 413


# =====================================================
# 12. RUN APPLICATION
# =====================================================

if __name__ == "__main__":
    app.run(debug=True)