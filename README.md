# 🚁 AISA – AI-Based Landmine Detection System

### Deep Learning | Computer Vision | Image Processing | AI-Based Terrain Classification

## 📌 Project Overview

**AISA** is an AI-based landmine detection research project designed to explore the use of deep learning and computer vision for identifying potentially hazardous terrain through image classification.

The system uses trained Convolutional Neural Network (CNN) architectures and ResNet-18 to analyse RGB and thermal imagery and classify uploaded images into two categories:

- **Free Zone:** Image classified as a potentially non-hazardous zone.
- **Mine-Related:** Image classified as potentially associated with mine-related terrain.

The project integrates PyTorch-based deep learning models with a Flask web application, providing an interface for image upload, preprocessing, classification and model performance evaluation.

The broader objective is to explore AI-assisted approaches that could support safer terrain assessment and reduce the need for direct human exposure to potentially hazardous environments.

## 🎯 Project Objectives

- Explore deep learning techniques for image-based landmine classification.
- Develop CNN-based image classification models.
- Implement ResNet-18 for RGB image classification.
- Explore RGB and thermal image processing.
- Integrate trained models into a Flask-based application.
- Evaluate classification performance using standard machine learning metrics.
- Investigate hardware acceleration to improve model inference efficiency.

## ⚙️ Key Features

### 1. RGB Image Classification
- Processes uploaded RGB images.
- Uses trained CNN and ResNet-18 architectures.
- Resizes images to 224 × 224 pixels.
- Applies image normalization.
- Classifies images into Free Zone and Mine-Related categories.

### 2. Thermal Image Processing
- Includes a thermal CNN model checkpoint.
- Supports thermal image preprocessing in the earlier CNN application.
- Explores thermal imagery as an additional source of visual information.

**Note:** Thermal inference is not currently enabled in the updated ResNet-18 application because its original model architecture needs to be restored.

### 3. Deep Learning Integration
- PyTorch-based model implementation.
- ResNet-18 architecture.
- Custom CNN architecture.
- Trained model checkpoint loading.
- Probability-based image classification.

### 4. Model Performance Evaluation

The application includes evaluation using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Evaluation uses labelled test data rather than relying only on training performance.

### 5. Web-Based Interface
- Flask backend.
- Image upload functionality.
- Image preprocessing and classification.
- Prediction confidence display.
- Model performance interface.

## 🧠 Deep Learning Architectures

The project contains multiple trained model implementations.

| Model | Purpose |
|---|---|
| Custom CNN | RGB and thermal image classification |
| ResNet-18 | RGB image classification |
| Thermal CNN | Thermal image classification research |

### ResNet-18 Configuration

The ResNet-18 implementation uses:

- Input resolution: 224 × 224
- Number of classes: 2
- Architecture: ResNet-18
- Framework: PyTorch
- Normalization: ImageNet mean and standard deviation
- Classification threshold: 0.46 for the free-zone class

## 🏗️ System Architecture

The project follows a modular architecture consisting of the following components:

**1. Data Processing Layer**
- RGB image loading.
- Image resizing.
- Image normalization.
- Thermal image preprocessing.
- Dataset preparation and labelling.

**2. Deep Learning Layer**
- CNN architecture.
- ResNet-18 architecture.
- Trained model loading.
- Image classification.
- Prediction probability calculation.

**3. Backend Layer**
- Flask application.
- Image upload handling.
- Model inference.
- Performance evaluation APIs.

**4. Presentation Layer**
- Web-based image upload interface.
- Classification results.
- Prediction confidence.
- Model performance information.

## 🔄 Working Principle

1. The user uploads an RGB or supported thermal image through the web interface.
2. The Flask backend validates the uploaded image.
3. The image undergoes preprocessing and normalization.
4. The processed image is converted into a PyTorch tensor.
5. The trained deep learning model performs inference.
6. The system generates a classification result and associated confidence score.
7. The results are displayed through the web application.
8. Model performance can be evaluated using labelled test datasets.

## 🛠️ Technology Stack

### Programming Languages
- Python

### Deep Learning & Machine Learning
- PyTorch
- Torchvision
- CNN
- ResNet-18
- Scikit-learn

### Image Processing
- Pillow
- NumPy

### Backend Development
- Flask
- REST APIs

### Data Processing
- Pandas
- NumPy
- CSV-based dataset management

### Development Tools
- VS Code
- Python
- Git
- GitHub

## 💻 Hardware Acceleration

The project explores computational acceleration for deep learning inference.

The original CNN application supports automatic device selection:

- **CPU:** Used as the fallback processing device.
- **GPU:** Uses CUDA when a compatible NVIDIA GPU is available.

The updated ResNet-18 application currently uses CPU execution.

GPU performance evaluation and TPU deployment should be documented separately once their implementations and measurements are verified.

## 📂 Project Structure

The repository is organised into the following logical modules:

```text
AISA-Landmine-Detection/
│
├── app.py
├── aisa_app_updated.py
│
├── models/
│   ├── resnet18_landmine_balanced.pth
│   ├── rgb_cnn_model.pth
│   ├── thermal_cnn_model.pth
│   └── resnet18_config.json
│
├── datasets/
│   ├── RGB/
│   ├── Thermal/
│   └── metadata/
│
├── results/
│   ├── rgb_preprocessed.csv
│   ├── dataset_labeled.csv
│   └── dataset_split.csv
│
├── requirements.txt
├── README.md
├── .gitignore
└── .gitattributes
```

*Note: This is a logical representation. Preserve the actual directory paths required by the Python applications.*

## 🚀 Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/Ayush18Shah/AISA-Landmine-Detection.git
```

### Step 2: Navigate to the Project

```bash
cd AISA-Landmine-Detection
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

Ensure the environment includes Flask, NumPy, Pandas, PyTorch, Torchvision, Pillow and Scikit-learn.

### Step 5: Run the Application

For the CNN implementation:

```bash
python app.py
```

For the updated ResNet-18 implementation:

```bash
python aisa_app_updated.py
```

Open the local Flask address displayed in the terminal.

**Important:** The model checkpoints and dataset directories must be placed where the selected application's path configuration expects them.

## 📊 Model Evaluation

The application supports evaluation through:

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Precision | Correctness of positive predictions |
| Recall | Ability to identify positive samples |
| F1-score | Balance between precision and recall |
| Confusion Matrix | Comparison of actual and predicted classes |

Actual evaluation results should be reported after running the model against the complete held-out test dataset.

## 🔬 Research and Future Enhancements

- Restoration and integration of the thermal CNN architecture.
- Further experimentation with RGB and thermal data fusion.
- GPU-based inference optimisation.
- Investigation of TPU or edge-AI deployment.
- Integration with drone-based image acquisition.
- LiDAR-based terrain analysis.
- Real-time image processing.
- Expanded dataset validation and model benchmarking.
- Improved model generalisation across different terrain conditions.

## ⚠️ Research Disclaimer

AISA is an experimental academic research prototype for image classification.

Its predictions do not establish whether landmines are actually present or absent. The system has not been validated or certified for operational mine detection, demining or field safety decisions.

It must not be used as a substitute for professional explosive ordnance detection equipment, trained personnel or established safety procedures.

## 🎓 Project Domain

- Artificial Intelligence
- Deep Learning
- Computer Vision
- Image Processing
- Embedded AI Research
- Hardware Acceleration
- Robotics and Drone-Based Applications

## 👨‍💻 Developed By

**Ayush Shah**

B.Tech – Computer Science and Engineering  
Specialization: Artificial Intelligence and Data Science

## 📄 License

Specify an appropriate license before allowing public reuse or redistribution of the source code, datasets and trained models.
