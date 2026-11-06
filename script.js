
"use strict";

// =====================================================
// 1. ELEMENTS AND STATE
// =====================================================

const input = document.getElementById("imageInput");
const dropZone = document.getElementById("dropZone");
const preview = document.getElementById("imagePreview");
const previewWrap = document.getElementById("previewWrap");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const analyzeButton = document.getElementById("analyzeButton");

let selectedFile = null;
let previewUrl = null;
let requestInProgress = false;

const maxSize = 10 * 1024 * 1024;
const validTypes = ["image/jpeg", "image/png", "image/bmp"];

// =====================================================
// 2. SAFE ELEMENT HELPERS
// =====================================================

function element(id) {
    return document.getElementById(id);
}

function setText(id, value) {
    const el = element(id);
    if (el) el.textContent = value;
}

function setStatus(title, message, progress) {
    setText("statusTitle", title);
    setText("statusMessage", message);
    setText("progressValue", progress + "%");

    const bar = element("progressFill");
    if (bar) bar.style.width = progress + "%";
}

function setStep(id, state, message) {
    const step = element(id);
    if (!step) return;

    step.classList.toggle("done", state === "done");

    const small = step.querySelector("small");
    const paragraph = step.querySelector("div p");

    if (small) small.textContent = state.toUpperCase();
    if (paragraph) paragraph.textContent = message;
}

function resetSteps() {
    setStep("stepUpload", "pending", "Waiting for image");
    setStep("stepPreprocess", "pending", "Not started");
    setStep("stepModel", "pending", "Model not connected");
    setStep("stepResult", "pending", "Awaiting model output");
}

function formatSize(bytes) {
    if (bytes < 1024 * 1024) {
        return (bytes / 1024).toFixed(1) + " KB";
    }

    return (bytes / (1024 * 1024)).toFixed(2) + " MB";
}

// =====================================================
// 3. IMAGE PREVIEW HELPERS
// =====================================================

function setPipelineImage(imageId, statusId, src, statusText) {
    const image = element(imageId);
    const status = element(statusId);

    if (image && src) {
        image.src = src;
        image.hidden = false;

        const icon = image.parentElement.querySelector("b");
        if (icon) icon.hidden = true;
    }

    if (status) status.textContent = statusText;
}

function resetPipeline() {
    [
        "pipelineOriginal",
        "pipelineGrayscale",
        "pipelineFiltered"
    ].forEach(function (id) {
        const image = element(id);
        if (!image) return;

        image.removeAttribute("src");
        image.hidden = true;

        const icon = image.parentElement.querySelector("b");
        if (icon) icon.hidden = false;
    });

    setText("pipelineOriginalStatus", "WAITING");
    setText("pipelineGrayscaleStatus", "WAITING");
    setText("pipelineFilteredStatus", "WAITING");
    setText("pipelineFeatureStatus", "WAITING");

    setText(
        "pipelineFeatureDescription",
        "Waiting for model analysis"
    );
}

// Generate honest visual previews in the browser.
// These are illustrative preprocessing previews;
// the model still receives the original uploaded image.
function createImagePreviews(file) {
    return new Promise(function (resolve, reject) {
        const image = new Image();

        image.onload = function () {
            try {
                const canvas = document.createElement("canvas");
                const ctx = canvas.getContext("2d");

                const width = Math.min(image.naturalWidth, 1200);
                const scale = width / image.naturalWidth;
                const height = Math.round(image.naturalHeight * scale);

                canvas.width = width;
                canvas.height = height;

                ctx.drawImage(image, 0, 0, width, height);

                const originalUrl = canvas.toDataURL("image/jpeg", 0.9);
                const originalPixels = ctx.getImageData(
                    0, 0, width, height
                );

                // Grayscale preview
                const grayCanvas = document.createElement("canvas");
                grayCanvas.width = width;
                grayCanvas.height = height;

                const grayCtx = grayCanvas.getContext("2d");
                const grayPixels = new ImageData(
                    new Uint8ClampedArray(originalPixels.data),
                    width,
                    height
                );

                for (let i = 0; i < grayPixels.data.length; i += 4) {
                    const gray =
                        0.299 * grayPixels.data[i] +
                        0.587 * grayPixels.data[i + 1] +
                        0.114 * grayPixels.data[i + 2];

                    grayPixels.data[i] = gray;
                    grayPixels.data[i + 1] = gray;
                    grayPixels.data[i + 2] = gray;
                }

                grayCtx.putImageData(grayPixels, 0, 0);

                // A simple box-blur preview, not the model's
                // actual internal feature representation.
                const filteredCanvas = document.createElement("canvas");
                filteredCanvas.width = width;
                filteredCanvas.height = height;

                const filteredCtx = filteredCanvas.getContext("2d");
                filteredCtx.filter = "blur(1px)";
                filteredCtx.drawImage(grayCanvas, 0, 0);

                resolve({
                    original: originalUrl,
                    grayscale: grayCanvas.toDataURL("image/jpeg", 0.9),
                    filtered: filteredCanvas.toDataURL("image/jpeg", 0.9)
                });
            } catch (error) {
                reject(error);
            } finally {
                URL.revokeObjectURL(image.src);
            }
        };

        image.onerror = function () {
            URL.revokeObjectURL(image.src);
            reject(new Error("Could not create image previews."));
        };

        image.src = URL.createObjectURL(file);
    });
}

async function updatePipeline(file, result) {
    try {
        const previews = await createImagePreviews(file);

        setPipelineImage(
            "pipelineOriginal",
            "pipelineOriginalStatus",
            previews.original,
            "COMPLETE"
        );

        setPipelineImage(
            "pipelineGrayscale",
            "pipelineGrayscaleStatus",
            previews.grayscale,
            "COMPLETE"
        );

        setPipelineImage(
            "pipelineFiltered",
            "pipelineFilteredStatus",
            previews.filtered,
            "PREVIEW"
        );

        setText("pipelineFeatureStatus", "CLASSIFIED");

        setText(
            "pipelineFeatureDescription",
            "ResNet-18 classification completed; no localized regions generated"
        );
    } catch (error) {
        console.error("Preview error:", error);
        setText("pipelineFeatureStatus", "UNAVAILABLE");
        setText(
            "pipelineFeatureDescription",
            "Image preview could not be generated"
        );
    }
}

// =====================================================
// 4. RESULT DISPLAY
// =====================================================

function resetResult() {
    const status = element("predictionStatus");

    if (status) {
        status.replaceChildren();

        const dot = document.createElement("span");
        dot.className = "dot";

        const label = document.createElement("strong");
        label.textContent = "Not available";

        status.append(dot, label);
    }

    setText("predictionValue", "—");
    setText("confidenceValue", "—");
    setText("regionValue", "—");
    setText("modelNoticeTitle", "Ready for analysis");
    setText(
        "modelNoticeText",
        "Upload an image to run the trained model."
    );

    const visual = element("resultVisual");
    if (visual) {
        visual.replaceChildren();

        const box = document.createElement("div");
        box.className = "result-empty";

        const symbol = document.createElement("div");
        symbol.textContent = "⌕";

        const heading = document.createElement("h3");
        heading.textContent = "No detection output yet";

        const paragraph = document.createElement("p");
        paragraph.textContent =
            "Upload an image to run the trained classifier.";

        box.append(symbol, heading, paragraph);
        visual.appendChild(box);
    }
}

function showPrediction(data, file) {
    setText(
        "predictionValue",
        data.prediction || "Prediction unavailable"
    );

    const score = Number(data.model_score);

    setText(
        "confidenceValue",
        Number.isFinite(score)
            ? score.toFixed(2) + "%"
            : "Not provided"
    );

    setText(
        "regionValue",
        "Not generated by classification model"
    );

    setText("modelNoticeTitle", "Analysis complete");

    setText(
        "modelNoticeText",
        data.message || "Image classification completed."
    );

    const status = element("predictionStatus");

    if (status) {
        status.replaceChildren();

        const dot = document.createElement("span");
        dot.className = "dot";

        const label = document.createElement("strong");
        label.textContent = "Prediction received";

        status.append(dot, label);
    }

    const visual = element("resultVisual");

    if (visual) {
        visual.replaceChildren();

        const image = document.createElement("img");
        image.alt = "Image analyzed by the classifier";
        image.src = data.image_url || previewUrl;

        visual.appendChild(image);
    }
}

// =====================================================
// 5. SELECT IMAGE
// =====================================================

function selectImage(file) {
    if (!file) return;

    if (!validTypes.includes(file.type)) {
        alert("Please select a JPG, JPEG, or PNG image.");
        return;
    }

    if (file.size > maxSize) {
        alert("The image must be smaller than 10 MB.");
        return;
    }

    if (previewUrl) URL.revokeObjectURL(previewUrl);

    selectedFile = file;
    previewUrl = URL.createObjectURL(file);

    preview.src = previewUrl;
    previewWrap.hidden = false;

    fileName.textContent = file.name;
    fileSize.textContent = formatSize(file.size);

    analyzeButton.disabled = false;

    setStatus("Image selected", "Ready to run AI analysis.", 0);

    resetSteps();
    resetResult();
    resetPipeline();

    setStep("stepUpload", "done", "Image selected");

    const visual = element("resultVisual");

    if (visual) {
        visual.replaceChildren();

        const image = document.createElement("img");
        image.src = previewUrl;
        image.alt = "Uploaded image awaiting analysis";

        visual.appendChild(image);
    }
}

input.addEventListener("change", function () {
    if (this.files && this.files.length) {
        selectImage(this.files[0]);
    }

    this.value = "";
});

dropZone.addEventListener("click", function (event) {
    if (event.target.closest("label")) return;
    input.click();
});

dropZone.addEventListener("dragover", function (event) {
    event.preventDefault();
    dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", function () {
    dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", function (event) {
    event.preventDefault();
    dropZone.classList.remove("drag-over");

    if (event.dataTransfer.files.length) {
        selectImage(event.dataTransfer.files[0]);
    }
});

// =====================================================
// 6. REMOVE IMAGE
// =====================================================

element("removeImage").addEventListener("click", function () {
    if (requestInProgress) return;

    selectedFile = null;

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }

    preview.removeAttribute("src");
    previewWrap.hidden = true;

    fileName.textContent = "No image selected";
    fileSize.textContent = "Choose an image to continue";

    analyzeButton.disabled = true;

    setStatus("Awaiting image", "Upload an image to begin.", 0);

    resetSteps();
    resetResult();
    resetPipeline();
});

// =====================================================
// 7. SAFE JSON REQUEST HELPER
// =====================================================

async function fetchJSON(url, options) {
    const response = await fetch(url, options);
    const contentType = response.headers.get("content-type") || "";

    if (!contentType.includes("application/json")) {
        const body = await response.text();

        console.error(
            "Expected JSON from",
            url,
            "but received:",
            body.slice(0, 300)
        );

        throw new Error(
            "The server returned HTML instead of JSON for " +
            url +
            ". Check the Flask route and terminal."
        );
    }

    const data = await response.json();

    if (!response.ok) {
        throw new Error(
            data.message || data.error || "Server request failed."
        );
    }

    return data;
}

// =====================================================
// 8. ANALYZE IMAGE
// =====================================================

analyzeButton.addEventListener("click", async function () {
    if (!selectedFile || requestInProgress) return;

    requestInProgress = true;

    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analyzing...";

    setStatus(
        "Uploading image",
        "Sending image to the trained ResNet-18 model.",
        20
    );

    setStep("stepUpload", "done", "Image sent to backend");
    setStep("stepPreprocess", "pending", "Preparing image");
    setStep("stepModel", "pending", "Running trained model");
    setStep("stepResult", "pending", "Waiting for prediction");

    const formData = new FormData();
    formData.append("image", selectedFile);
    formData.append("model_type", "rgb");

    try {
        const data = await fetchJSON("/analyze", {
            method: "POST",
            body: formData
        });

        if (data.success !== true) {
            throw new Error(
                data.message || "Image analysis failed."
            );
        }

        if (data.model_connected === false) {
            throw new Error("The trained model is not connected.");
        }

        setStatus(
            "Generating previews",
            "Displaying image processing previews.",
            55
        );

        setStep(
            "stepPreprocess",
            "done",
            "Image resized and normalized by backend"
        );

        await updatePipeline(selectedFile, data);

        setStatus(
            "Analysis complete",
            "The trained model returned a classification.",
            100
        );

        setStep("stepUpload", "done", "Image uploaded successfully");
        setStep("stepPreprocess", "done", "Previews generated");
        setStep(
            "stepModel",
            "done",
            data.model_name || "ResNet-18"
        );
        setStep("stepResult", "done", "Prediction received");

        showPrediction(data, selectedFile);

    } catch (error) {
        console.error("Image analysis failed:", error);

        setStatus("Analysis failed", error.message, 0);

        setStep("stepModel", "pending", "Analysis failed");
        setStep("stepResult", "pending", "No prediction received");

        setText("modelNoticeTitle", "Analysis failed");
        setText("modelNoticeText", error.message);

        alert(error.message);

    } finally {
        requestInProgress = false;
        analyzeButton.disabled = !selectedFile;
        analyzeButton.innerHTML = 'Analyze Image <span>→</span>';
    }
});

// =====================================================
// 9. MODEL PERFORMANCE
// =====================================================

async function loadModelPerformance() {
    const section = element("performance");
    if (!section) return;

    const metricCards = section.querySelectorAll(".metric-card");

    try {
        metricCards.forEach(function (card) {
            const value = card.querySelector("strong");
            if (value) value.textContent = "Loading...";
        });

        // This endpoint must be implemented in Flask and
        // must calculate metrics from labelled test images.
        const data = await fetchJSON("/model-performance");

        if (data.success !== true) {
            throw new Error(
                data.message || "Evaluation data is unavailable."
            );
        }

        const metricValues = [
            data.accuracy,
            data.precision,
            data.recall,
            data.f1_score
        ];

        metricCards.forEach(function (card, index) {
            const value = card.querySelector("strong");
            const metric = Number(metricValues[index]);

            if (value) {
                value.textContent = Number.isFinite(metric)
                    ? metric.toFixed(2) + "%"
                    : "Unavailable";
            }
        });

        const description = section.querySelector(
            ".section-heading .subtext"
        );

        if (description) {
            description.textContent =
                "Evaluation on " + data.test_samples +
                " labelled test images.";
        }

        const badge = section.querySelector(".step-label");
        if (badge) badge.textContent = "TEST SET EVALUATED";

        drawPerformanceChart(section, metricValues);

    } catch (error) {
        console.error("Performance loading failed:", error);

        metricCards.forEach(function (card) {
            const value = card.querySelector("strong");
            if (value) value.textContent = "Unavailable";
        });

        const description = section.querySelector(
            ".section-heading .subtext"
        );

        if (description) {
            description.textContent =
                "No evaluation results received. " +
                "The Flask evaluation endpoint and labelled test dataset are required.";
        }

        const badge = section.querySelector(".step-label");
        if (badge) badge.textContent = "TEST DATA REQUIRED";

        const chart = section.querySelector(".chart-empty");

        if (chart) {
            chart.replaceChildren();

            const heading = document.createElement("strong");
            heading.textContent = "Evaluation data unavailable";

            const paragraph = document.createElement("p");
            paragraph.textContent =
                "Connect the labelled test dataset to the backend evaluation endpoint.";

            chart.append(heading, paragraph);
        }
    }
}

function drawPerformanceChart(section, values) {
    const chart = section.querySelector(".chart-empty");
    if (!chart) return;

    const names = ["Accuracy", "Precision", "Recall", "F1 Score"];
    chart.replaceChildren();

    const bars = document.createElement("div");

    bars.style.cssText =
        "display:flex;align-items:flex-end;justify-content:center;" +
        "gap:20px;height:200px;padding:20px;box-sizing:border-box;";

    values.forEach(function (rawValue, index) {
        const value = Number(rawValue);

        if (!Number.isFinite(value)) return;

        const column = document.createElement("div");
        column.style.cssText =
            "display:flex;flex-direction:column;align-items:center;" +
            "justify-content:flex-end;gap:8px;flex:1;height:100%;";

        const label = document.createElement("strong");
        label.textContent = value.toFixed(2) + "%";

        const bar = document.createElement("div");
        bar.style.cssText =
            "width:100%;max-width:70px;background:#27d6a1;" +
            "border-radius:6px 6px 0 0;";

        bar.style.height = Math.max(2, Math.min(100, value)) + "%";

        const name = document.createElement("small");
        name.textContent = names[index];

        column.append(label, bar, name);
        bars.appendChild(column);
    });

    const heading = document.createElement("h4");
    heading.textContent = "Test Set Evaluation";

    chart.append(bars, heading);
}

// =====================================================
// 10. NAVIGATION
// =====================================================

document.querySelectorAll(".nav-link").forEach(function (link) {
    link.addEventListener("click", function () {
        document.querySelectorAll(".nav-link").forEach(function (item) {
            item.classList.remove("active");
        });

        this.classList.add("active");
    });
});

// =====================================================
// 11. INITIALIZE
// =====================================================

document.addEventListener("DOMContentLoaded", function () {
    resetSteps();
    resetPipeline();
    loadModelPerformance();
});
