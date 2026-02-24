# 🌿 Leaf Disease Detection using CNN + Vision Transformer

This project implements a deep learning–based system for classifying common wheat leaf diseases using a hybrid **Convolutional Neural Network (CNN)** and **Vision Transformer (ViT)** architecture. It provides a complete pipeline—from dataset preparation and model training to evaluation, Grad-CAM visualization, and deployment via a Flask web application.

---

## 📚 Table of Contents

- [📦 Dataset](#-dataset)
- [🧠 Model Architecture](#-model-architecture)
- [📁 Project Structure](#-project-structure)
- [⚙️ Installation](#️-installation)
- [🚀 Usage](#-usage)
- [🌐 Web Application](#-web-application)
- [💾 Saved Models](#-saved-models)
- [🛠 Features](#-features)
- [🧩 Dependencies](#-dependencies)
- [🧪 Examples](#-examples)
- [🐞 Troubleshooting](#-troubleshooting)
- [✍️ Author](#️-author)
- [📜 License](#-license)

---

## 📦 Dataset

The dataset contains labeled wheat leaf images categorized into the following five classes:

- **BlackPoint**
- **FusariumFootRot**
- **HealthyLeaf**
- **LeafBlight**
- **WheatBlast**

**Directory Structure:**
```bash
dataset1/
├── Train/
│   ├── BlackPoint/
│   ├── FusariumFootRot/
│   ├── HealthyLeaf/
│   ├── LeafBlight/
│   └── WheatBlast/
├── Validation/
│   ├── BlackPoint/
│   ├── FusariumFootRot/
│   ├── HealthyLeaf/
│   ├── LeafBlight/
│   └── WheatBlast/
└── Test/
    ├── BlackPoint/
    ├── FusariumFootRot/
    ├── HealthyLeaf/
    ├── LeafBlight/
    └── WheatBlast/
```

> ⚠️ **Note:** The dataset is not included in this repository.

### 🔗 Dataset Download

Download it from Kaggle:  
👉 [Wheat Leaf Disease Dataset](https://www.kaggle.com/datasets/khanaamer/wheat-leaf-disease-dataset)

After downloading, extract the files to the project root directory and ensure the structure matches the format above.

---

## 🧠 Model Architecture

This hybrid architecture leverages the strengths of both CNNs and Vision Transformers:

- **CNN:** Extracts low-level spatial features and acts as a tokenization layer.
- **ViT:** Captures global dependencies using self-attention mechanisms.
- **Output:** Multi-class classification via softmax.

**Task Type:** Multi-class image classification  
**Domain:** Wheat leaf disease detection

---

## 📁 Project Structure

```
LEAF_DISEASE/
├── dataset1/                  # Dataset directory (not included in Git)
├── models/
│   └── cnn_vit_model.py       # CNN + ViT architecture
├── preprocessing/             # (Optional) Dataset prep scripts
├── saved_models/
│   ├── vit_dataset-1.h5       # Trained model
│   └── history_vit_dataset-1.pkl
├── static/                    # Static files (for Flask)
├── templates/                 # HTML templates (for Flask)
├── utils/                     # Utility functions
├── app.py                     # Flask web app
├── main.py                    # Model training script
├── predict_image.py           # Inference for a single image
├── evaluate_vitmodel.py       # Evaluation script
├── gradcam_vit.py             # Grad-CAM visualizer
├── plot_history.py            # Plot training history
├── requirements.txt           # Dependency file
└── README.md
```

---

## ⚙️ Installation

### ✅ Requirements

- Python 3.10.11
- OS: Windows / Linux / macOS
- Dependencies listed in `requirements.txt`

### 🔧 Steps

1️⃣ **Clone the Repository**

```bash
git clone <repository-url>
cd LEAF_DISEASE
```

2️⃣ **Create Virtual Environment (Recommended)**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

3️⃣ **Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 🔧 Train the model

```bash
python main.py
```

### 🧪 Evaluate the model

```bash
python evaluate_vitmodel.py
```

### 📸 Predict a single image

```bash
python predict_image.py
```

### 🔍 Grad-CAM Visualization

```bash
python gradcam_vit.py
```

### 🌐 Run Web Application

```bash
python app.py
```

Then open in your browser:  
[http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 💾 Saved Models

Pre-trained model and logs are included:

- `vit_dataset-1.h5` — Trained CNN-ViT model
- `history_vit_dataset-1.pkl` — Accuracy/loss history

✅ These files allow you to perform predictions without retraining.

---

## 🛠 Features

- ✅ Hybrid CNN + Vision Transformer for improved accuracy
- ✅ Grad-CAM visualizations to interpret predictions
- ✅ Clean training, evaluation, and prediction pipelines
- ✅ Web app interface for real-time inference
- ✅ Modular, extensible codebase

---

## 🧩 Dependencies

All required packages are listed in `requirements.txt`.

To install:

```bash
pip install -r requirements.txt
```

Key dependencies include:

- TensorFlow / Keras
- NumPy, Matplotlib
- OpenCV
- Flask
- scikit-learn
- PIL (Pillow)

---

## 🧪 Examples

- 📷 Upload a wheat leaf image via the web UI.
- ⚙️ The model predicts and displays the disease class.
- 🔬 Use Grad-CAM to highlight affected regions in the image.

---

## 🐞 Troubleshooting

- **Web app not launching?** Ensure Flask is installed and port 5000 is free.
- **Model not found?** Check `saved_models/` directory or retrain using `main.py`.
- **Incorrect predictions?** Verify that input images match training image dimensions and format.

---



---

## 📜 License

This project is intended **solely for academic and research purposes**.  
For commercial use, please contact the author.
