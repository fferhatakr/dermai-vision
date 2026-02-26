# 🩺 Skin Cancer Detection - Dermatologist in Your Pocket (v6.1 - End-to-End Web System)

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Status](https://img.shields.io/badge/Status-Under_Development-green.svg)

## ⚠️ Disclaimer:
**This project is an AI research and engineering demonstration.**
**It is NOT intended for real medical diagnosis.**


This project is an end-to-end deep learning-based skin cancer classification and retrieval assistant. It covers a complete engineering journey: starting from flat-layer models, extending to custom CNNs, integrating **Multimodal Fusion (MobileNetV3 & DistilBERT)**, and finally evolving into a **Content-Based Image Retrieval (CBIR)** system served via a modern REST API and Web Interface.

## 🌟 What's New in v6.1: End-to-End Web System
The project is no longer just a set of training scripts. It is now a fully functional product:
* **The Brain (KNN Retrieval):** Instead of standard classification, the model extracts 576-dimensional feature vectors (embeddings) from a new patient's image and compares them against a vast, pre-calculated database of diagnosed cases using K-Nearest Neighbors.
* **The Backend (FastAPI):** A lightning-fast REST API (`uvicorn`) that handles image processing, tensor normalization, and real-time similarity matching.
* **The Frontend (Streamlit):** An interactive, user-friendly web interface where users can upload dermoscopy images and receive instant, confidence-based diagnostic feedback.

---
# 🚀 Engineering Journey & Model Evolution

The project was developed step by step, with each version improving the model's real-world data adaptation and deployment readiness.

### 🚀 Model Evolution and Performance Table

The project was developed step by step, with each version improving the model's real-world data adaptation and deployment readiness.

### Vision Models (Image Analysis)

| Version | Architecture | Technique | Test Accuracy | Avg. Loss | Key Improvement |
| :--- | :--- | :--- | :--- | :--- | :--- |
| v1 | Linear (MLP) | Baseline | 68.83% | 0.9014 | Basic skeleton established. |
| v3.1 | Custom CNN | Class Weights | 49.58% | 1.1857 | Overfitting broken; diagnostic blindness for rare classes eliminated. |
| v4.0 | ResNet18 | Full Retraining | 78.75% | 0.7465 | Pre-trained ImageNet weights integrated; large jump in lesion feature understanding. |
| v4.2 | MobileNetV3-Small | Mobile Optimization | 77.17% | 0.1982 | Best-model checkpointing added; lightweight architecture optimized for on-device inference. |
| v5.2 | MobileNetV3 | PyTorch Lightning | — | — | Training pipeline refactored; `ReduceLROnPlateau` scheduler added for dynamic LR adjustment. |
| v6.0 | MobileNetV3 + Triplet | Visual Similarity Search | — | 0.046 | **The Big Pivot:** Transitioned from classification to similarity learning. MLOps (YAML configs) and unit testing integrated for a robust production pipeline. |


### v5.2 - Lightning & Optimization Update
* ⚡ **Training Pipeline Refactored:** Migrated from Vanilla PyTorch to PyTorch Lightning for scalable and clean training architecture.       
* 📉 **Smart Optimization:** Integrated `ReduceLROnPlateau` scheduler for dynamic learning rate adjustments to prevent overfitting.     


> **Engineering Note (v4.2):** Hitting ~77% accuracy with a lightweight model like MobileNetV3-Small on a highly imbalanced, 7-class medical dataset is a massive optimization milestone. The model is now perfectly sized to be converted into TorchScript for native iOS (Swift) deployment without draining device resources.

### NLP Models (Symptom Analysis)

| Version | Architecture | Dataset | Accuracy | Key Improvement |
| :--- | :--- | :--- | :--- | :--- |
| v1.0 | DistilBERT (EN) | Custom Dataset | 96.08% | Semantic risk factor detection from patient-reported text. |

> **Multimodal Fusion Note:** The project incorporates not only image pixels but also free-text patient complaints (e.g., *"rapid growth"*, *"bleeding"*), computing a **Hybrid Score** to improve overall diagnostic accuracy.


## 📂 File Structure

```text
AI_DET_PROJECT/
├── configs/
│   ├── database_config.yaml
│   ├── inference_config.yaml
│   ├── model_config.yaml
│   └── train_config.yaml
├── Data/
│   ├── images/
│   └── metadata/
├── lightning_logs/
├── models/
├── notebooks/
├── src/
│   ├── datalar/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── nlp_dataset.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── hybrid_predict.py
│   │   ├── evaluate_retrieval.py
│   │   └── predict.py
│   ├── api/
│   │   └──main.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── nlp_model.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── nlp_train.py
│   │   ├── train_vanilla.py
│   │   ├── lightning_model.py
│   │   ├── train_lightning.py
│   │   ├── train_triplet.py
│   │   └──utils.py
│   ├──ui/
│   │   └──app.py
├── test/
│   ├── test_dataset.py
│   ├── test_inference.py
│   ├── test_model.py
│   └── test_nlp_dataset.py
├── .env
├── .gitignore
├── README.md
├── pytest.ini
└── requirements.txt
```

## 🛠️ Technologies and Techniques Used

- **Architectures: Custom CNNs, ResNet18, MobileNetV3**
- **Transfer Learning: Fine-tuning pre-trained ImageNet weights (requires_grad=True, low learning rate)**  
- **Data Pipeline: RandomHorizontalFlip, RandomRotation, ColorJitter, ImageNet Normalization.** 
- **Imbalanced Data Solution: Class Weights (sklearn) for vision; data augmentation for NLP.** 
- **Optimization: AdamW optimizer, Dynamic Learning Rate, Softmax Probability Scoring.**  

## 🎯 Roadmap

- [x] v2.0: Migration to CNN architecture.
- [x] v2.1: Improving model reliability with Data Augmentation.
- [x] v3.1: Solving the imbalanced data problem with Class Weights.
- [x] v4.0: Maximizing accuracy with Transfer Learning (ResNet18).
- [x] v4.2: Mobile optimization with MobileNetV3-Small.
- [x] v5.0: Multimodal NLP Integration (Symptom analysis).
- [x] v5.1: Unified Multimodal Fusion (Combining Image + Text scores).
- [x] v5.2: Training pipeline refactored with PyTorch Lightning & smart LR scheduling.
- [x] v6.0: The Big Pivot — Transitioned from classification to Visual Similarity Search (Triplet Loss). MLOps & unit testing integrated.
- [x] v6.1: End-to-End System integration (FastAPI backend + Streamlit UI) with KNN retrieval.
- [ ] v7.0: Cross-Platform Mobile Deployment (Android (Kotlin) / iOS (Swift)).


## ⚙️ Installation 

1. Clone the repo:

```bash
git clone https://github.com/fferhatakr/AI_DET_PROJECT.git
cd AI_DET_PROJECT
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

# 🚀 Running the Project

The project now operates as a full-stack application. You need to run both the backend (API) and the frontend (UI) simultaneously.

**1. Start the Backend API (FastAPI & KNN Engine):**
```bash
uvicorn src.api.main:app --reload
```

**2. Start the Web Interface (Streamlit):**
Open a new terminal window (keep the API running) and execute:
```bash
streamlit run src/ui/app.py
```

### 🛠️ Developer Guide (Training from Scratch)

**1. Train Vision Model (PyTorch Lightning & Triplet Loss)**
```bash
python src/training/train_lightning.py
```
**2. Extract and Save Embeddings (Update the KNN Database)**
```bash
python src/inference/evaluate_retrieval.py
```
**3. Train NLP Model (Symptom Analysis)**
```bash
python src/training/nlp_train.py
```



# 🧠 Multimodal Fusion (Hybrid Diagnosis)
```bash
python src/inference/hybrid_predict.py

# Output Example:
# 📸 Image Risk : %95.38
# ✍️ Complaint Risk : %99.92
# 🧠 HYBRID SCORE : %98.10
# 🩺 DIAGNOSIS : ⚠️ RISKY (Consult a Specialist)
```

### ✍️ NLP Inference (Symptom Analysis)
You can test the NLP model directly from your Python code:

```bash
from src.inference.predict import predict_symptom

# Analyze patient complaint:
result = predict_symptom("My lesion's color has darkened and it bleeds.")
print(f"Output: {result} Risky")
```



