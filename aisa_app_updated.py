
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image, UnidentifiedImageError
from flask import Flask, jsonify, render_template, request, url_for
from torchvision import models, transforms
from werkzeug.utils import secure_filename
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ==================================================
# 1. APPLICATION CONFIGURATION
# ==================================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    "png", "jpg", "jpeg", "bmp", "tif", "tiff"
}

DEVICE = torch.device("cpu")


# ==================================================
# 2. MODEL AND DATASET PATHS
# ==================================================

MODEL_DIR_CANDIDATES = [
    PROJECT_DIR / "DatainBrief_dataset_1" / "models",
    PROJECT_DIR / "AISA PROJECT" / "DatainBrief_dataset_1" / "models",
    BASE_DIR / "models",
]


def find_model_file(filename):
    for folder in MODEL_DIR_CANDIDATES:
        path = folder / filename
        if path.is_file():
            return path
    return None


RGB_MODEL_PATH = find_model_file(
    "resnet18_landmine_balanced.pth"
)

RGB_CONFIG_PATH = find_model_file(
    "resnet18_config.json"
)

THERMAL_MODEL_PATH = find_model_file(
    "thermal_cnn_model.pth"
)

DATASET_DIR = PROJECT_DIR / "DatainBrief_dataset_1"

CSV_PATH = DATASET_DIR / "results" / "rgb_preprocessed.csv"

TEST_IMAGE_DIR = (
    DATASET_DIR / "processed" / "rgb" / "test"
)

# Metrics are calculated once per server process.
_model_performance_cache = None


# ==================================================
# 3. RGB IMAGE PREPROCESSING
# ==================================================

RGB_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==================================================
# 4. LOAD RESNET-18 RGB MODEL
# ==================================================

rgb_model = None
rgb_model_error = None

FREE_ZONE_THRESHOLD = 0.46

CLASS_NAMES = {
    0: "free_zone",
    1: "mine_related"
}


def load_rgb_model():
    global rgb_model
    global rgb_model_error
    global FREE_ZONE_THRESHOLD
    global CLASS_NAMES

    if RGB_MODEL_PATH is None:
        rgb_model_error = (
            "ResNet-18 checkpoint was not found."
        )
        print("RGB MODEL ERROR:", rgb_model_error)
        return

    try:
        if RGB_CONFIG_PATH and RGB_CONFIG_PATH.exists():
            with open(
                RGB_CONFIG_PATH, "r", encoding="utf-8"
            ) as file:
                config = json.load(file)

            FREE_ZONE_THRESHOLD = float(
                config.get("free_zone_threshold", 0.46)
            )

            configured_names = config.get("class_names", {})

            if configured_names:
                CLASS_NAMES = {
                    int(key): value
                    for key, value in configured_names.items()
                }

        checkpoint = torch.load(
            RGB_MODEL_PATH,
            map_location=DEVICE,
            weights_only=False
        )

        rgb_model = models.resnet18(weights=None)

        rgb_model.fc = nn.Linear(
            rgb_model.fc.in_features, 2
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        rgb_model.load_state_dict(state_dict)
        rgb_model.to(DEVICE)
        rgb_model.eval()

        rgb_model_error = None

        print("RGB ResNet-18 loaded successfully.")
        print("RGB checkpoint:", RGB_MODEL_PATH)
        print("Classification threshold:", FREE_ZONE_THRESHOLD)

    except Exception as error:
        rgb_model = None
        rgb_model_error = str(error)
        print("RGB MODEL ERROR:", rgb_model_error)


load_rgb_model()


# ==================================================
# 5. THERMAL MODEL STATUS
# ==================================================
# Thermal inference remains disabled until its original
# trained architecture is restored.

thermal_model = None

thermal_model_error = (
    "Thermal model is disabled because its original "
    "architecture has not yet been restored."
)

if THERMAL_MODEL_PATH is None:
    thermal_model_error = (
        "Thermal model checkpoint was not found."
    )

print("Thermal model status:", thermal_model_error)


# ==================================================
# 6. VALIDATE UPLOADED FILES
# ==================================================

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ==================================================
# 7. RGB IMAGE PREDICTION
# ==================================================

def predict_rgb(image_path):

    if rgb_model is None:
        raise RuntimeError(
            "RGB model is unavailable: "
            + str(rgb_model_error)
        )

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = RGB_TRANSFORM(image)
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = rgb_model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    free_probability = float(probabilities[0].item())
    mine_probability = float(probabilities[1].item())

    # Use the configured validation-selected threshold.
    if free_probability >= FREE_ZONE_THRESHOLD:
        predicted_class = 0
    else:
        predicted_class = 1

    prediction = CLASS_NAMES.get(
        predicted_class,
        "free_zone" if predicted_class == 0
        else "mine_related"
    )

    confidence = float(
        probabilities[predicted_class].item()
    )

    return {
        "prediction": prediction,
        "predicted_class": predicted_class,
        "model_score": round(confidence * 100, 2),
        "free_zone_probability": round(
            free_probability * 100, 2
        ),
        "mine_related_probability": round(
            mine_probability * 100, 2
        )
    }


# ==================================================
# 8. HOME PAGE
# ==================================================

@app.route("/")
def home():
    return render_template("index.html")


# ==================================================
# 9. MODEL STATUS API
# ==================================================

@app.route("/model-status")
def model_status():

    return jsonify({
        "device": str(DEVICE),

        "rgb_model_loaded": rgb_model is not None,
        "rgb_model_file": (
            RGB_MODEL_PATH.name
            if RGB_MODEL_PATH else None
        ),
        "rgb_model_error": rgb_model_error,

        "thermal_model_loaded": False,
        "thermal_model_file": (
            THERMAL_MODEL_PATH.name
            if THERMAL_MODEL_PATH else None
        ),
        "thermal_model_error": thermal_model_error
    })


# ==================================================
# 10. MODEL PERFORMANCE API
# ==================================================

@app.route("/model-performance")
def model_performance():

    global _model_performance_cache

    # Return previously calculated metrics.
    if _model_performance_cache is not None:
        return jsonify(_model_performance_cache)

    try:
        if rgb_model is None:
            return jsonify({
                "success": False,
                "error": (
                    "The ResNet-18 model is not loaded: "
                    + str(rgb_model_error)
                )
            }), 503

        if not CSV_PATH.is_file():
            return jsonify({
                "success": False,
                "error": f"Dataset CSV not found: {CSV_PATH}"
            }), 500

        if not TEST_IMAGE_DIR.is_dir():
            return jsonify({
                "success": False,
                "error": (
                    "Test image directory not found: "
                    + str(TEST_IMAGE_DIR)
                )
            }), 500

        # Read dataset metadata.
        df = pd.read_csv(CSV_PATH)

        required_columns = {
            "processed_filepath",
            "mine_label",
            "split"
        }

        if not required_columns.issubset(df.columns):
            return jsonify({
                "success": False,
                "error": (
                    "CSV must contain processed_filepath, "
                    "mine_label, and split columns."
                )
            }), 500

        # Use only the held-out test split.
        test_df = df[
            df["split"].astype(str).str.strip().str.lower()
            == "test"
        ].copy()

        test_df = test_df.dropna(
            subset=["processed_filepath", "mine_label"]
        )

        if test_df.empty:
            return jsonify({
                "success": False,
                "error": "No labelled test rows were found."
            }), 500

        y_true = []
        y_pred = []
        missing_images = []
        invalid_rows = []

        for _, row in test_df.iterrows():

            # CSV paths belong to the original computer.
            # Match the recorded filename to our local test folder.
            filename = Path(
                str(row["processed_filepath"]).replace("\\", "/")
            ).name

            image_path = TEST_IMAGE_DIR / filename

            if not image_path.is_file():
                missing_images.append(filename)
                continue

            try:
                actual_label = int(row["mine_label"])

                if actual_label not in (0, 1):
                    invalid_rows.append(filename)
                    continue

                result = predict_rgb(image_path)

                y_true.append(actual_label)
                y_pred.append(
                    int(result["predicted_class"])
                )

            except (ValueError, TypeError, OSError,
                    UnidentifiedImageError) as image_error:

                invalid_rows.append(filename)

                print(
                    "EVALUATION IMAGE ERROR:",
                    filename,
                    repr(image_error)
                )

        # Do not silently report metrics if files are missing
        # or the full labelled test set could not be evaluated.
        if missing_images or invalid_rows:
            return jsonify({
                "success": False,
                "error": (
                    "The complete labelled test set could not "
                    "be evaluated. Check the file matching "
                    "and image errors before reporting metrics."
                ),
                "expected_test_rows": int(len(test_df)),
                "evaluated_images": int(len(y_true)),
                "missing_images_count": len(missing_images),
                "missing_images_sample": missing_images[:10],
                "invalid_images_count": len(invalid_rows),
                "invalid_images_sample": invalid_rows[:10]
            }), 500

        if len(y_true) != len(test_df):
            return jsonify({
                "success": False,
                "error": (
                    "The number of evaluated images does not "
                    "match the number of labelled test rows."
                ),
                "expected_test_rows": int(len(test_df)),
                "evaluated_images": int(len(y_true))
            }), 500

        # Positive class: mine_related (label 1).
        accuracy = accuracy_score(y_true, y_pred)

        precision = precision_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0
        )

        # Rows = actual class; columns = predicted class.
        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        )

        _model_performance_cache = {
            "success": True,
            "model_name": "ResNet-18",
            "test_samples": len(y_true),

            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1_score": round(f1 * 100, 2),

            "positive_class": "mine_related",
            "class_names": [
                "free_zone",
                "mine_related"
            ],

            "confusion_matrix": matrix.tolist(),
            "missing_images": 0,
            "invalid_images": 0
        }

        print("MODEL PERFORMANCE:", _model_performance_cache)

        return jsonify(_model_performance_cache)

    except Exception as error:
        print("MODEL PERFORMANCE ERROR:", repr(error))

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


# ==================================================
# 11. IMAGE ANALYSIS API
# ==================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "message": "Please upload an image."
        }), 400

    uploaded_file = request.files["image"]

    if uploaded_file.filename == "":
        return jsonify({
            "success": False,
            "message": "No image was selected."
        }), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify({
            "success": False,
            "message": "Unsupported image format."
        }), 400

    model_type = request.form.get(
        "model_type", "rgb"
    ).lower()

    if model_type not in {"rgb", "thermal"}:
        return jsonify({
            "success": False,
            "message": "Invalid model type."
        }), 400

    if model_type == "thermal":
        return jsonify({
            "success": False,
            "model_connected": False,
            "message": (
                "The thermal model is not available yet. "
                "Please select the RGB model."
            )
        }), 503

    safe_filename = secure_filename(uploaded_file.filename)

    if not safe_filename:
        return jsonify({
            "success": False,
            "message": "Invalid filename."
        }), 400

    unique_filename = (
        uuid.uuid4().hex + "_" + safe_filename
    )

    image_path = UPLOAD_FOLDER / unique_filename

    try:
        uploaded_file.save(str(image_path))

        # Verify the upload is a valid image.
        with Image.open(image_path) as check_image:
            check_image.verify()

        result = predict_rgb(image_path)

        image_url = url_for(
            "static",
            filename="uploads/" + unique_filename
        )

        return jsonify({
            "success": True,
            "image_url": image_url,

            "model_connected": True,
            "model_type": "rgb",
            "model_name": "ResNet-18",

            "prediction": result["prediction"],
            "predicted_class": result["predicted_class"],
            "model_score": result["model_score"],

            "free_zone_probability": (
                result["free_zone_probability"]
            ),

            "mine_related_probability": (
                result["mine_related_probability"]
            ),

            "message": "RGB image analysis completed."
        })

    except (UnidentifiedImageError, OSError):
        return jsonify({
            "success": False,
            "message": (
                "The uploaded file is not a readable image."
            )
        }), 400

    except Exception as error:
        print("ANALYSIS ERROR:", repr(error))

        return jsonify({
            "success": False,
            "model_connected": rgb_model is not None,
            "message": (
                "Image analysis failed. Check the Flask "
                "terminal for the detailed error."
            )
        }), 500


# ==================================================
# 12. START SERVER
# ==================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
