# 🩺 Skin Cancer Detection - Dermatologist in Your Pocket (v5.1 - Multimodal Fusion)

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Status](https://img.shields.io/badge/Status-Under_Development-green.svg)

## ⚠️ Disclaimer:
**This project is an AI research and engineering demonstration.**
**It is NOT intended for real medical diagnosis.**


This project is a deep learning-based skin cancer classification assistant developed using PyTorch. The project covers an engineering journey that starts from flat-layer models (Linear), extends to custom CNN architectures, and currently utilizes **Multimodal Fusion (MobileNetV3 & DistilBERT)** for mobile-optimized, high-accuracy predictions.

### 🚀 Model Evolution and Performance Table

The project was developed step by step, with each version improving the model's real-world data adaptation and deployment readiness.

### Vision Models (Image Analysis)

| Version | Architecture | Technique | Test Accuracy | Avg. Loss | Key Improvement |
| :--- | :--- | :--- | :--- | :--- | :--- |
| v1 | Linear (MLP) | Baseline | 68.83% | 0.9014 | Basic skeleton established. |
| v3.1 | Custom CNN | Class Weights | 49.58% | 1.1857 | Overfitting broken; diagnostic blindness for rare classes eliminated. |
| v4.0 | ResNet18 | Full Retraining | 78.75% | 0.7465 | Pre-trained ImageNet weights integrated; large jump in lesion feature understanding. |
| v4.2 | MobileNetV3-Small | Mobile Optimization | 77.17% | 0.1982 | Best-model checkpointing added; lightweight architecture optimized for on-device inference. |
| v5.2 | MobileNetV3-Small | PyTorch Lightning | — | — | Training pipeline refactored; `ReduceLROnPlateau` scheduler added for dynamic LR adjustment. |
| v6.0 | MobileNetV3 + Triplet | Visual Similarity Search | — | 0.046 | **The Big Pivot:** Transitioned from classification to similarity learning. MLOps (YAML configs) and unit testing integrated for a robust production pipeline. |


### v5.2 - Lightning & Optimization Update
* ⚡ **Training Pipeline Refactored:** Migrated from Vanilla PyTorch to PyTorch Lightning for scalable and clean training architecture.       
* 📉 **Smart Optimization:** Integrated `ReduceLROnPlateau` scheduler for dynamic learning rate adjustments to prevent overfitting.     


> **Engineering Note (v4.2):** Hitting ~77% accuracy with a lightweight model like MobileNetV3-Small on a highly imbalanced, 7-class medical dataset is a massive optimization milestone. The model is now perfectly sized to be converted into TorchScript for native iOS(swift) deployment without draining device resources.

### NLP Models (Symptom Analysis)

| Version | Architecture | Dataset | Accuracy | Key Improvement |
| :--- | :--- | :--- | :--- | :--- |
| v1.0 | DistilBERT (TR) | Custom Dataset | 96.08% | Semantic risk factor detection from patient-reported text. |

> **Note (v5.0):** The project has entered a **multimodal phase** — diagnostics now incorporate not only image pixels but also free-text patient complaints (e.g. *"rapid growth"*, *"bleeding"*), improving overall diagnostic accuracy.


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
│   │   └── predict.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── nlp_model.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── nlp_train.py
│   │   └── train_vanilla.py
│   ├── __init__.py
│   ├── lightning_model.py
│   ├── train_lightning.py
│   ├── train_triplet.py
│   └── utils.py
├── test/
│   ├── test_dataset.py
│   ├── test_inference.py
│   ├── test_lightning_model.py
│   ├── test_model.py
│   ├── test_nlp_dataset.py
│   └── test_utils.py
├── .env
├── .gitignore
├── README.md
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
- [ ] v7.0: Mobile Deployment (Android Kotlin integration).


## ⚙️ Installation 

1. Clone the repo:

```bash
git clone https://github.com/your_username/AI_DET_PROJECT.git
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

## 🚀 Running the Project

Since we now have a modular structure, you can start training directly from the terminal:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train NLP Model
python src/training/nlp_train.py

# 3. Train Vision Model (Lightning)
python src/train_lightning.py
```

## 🛠️ Using the Model in Code (Usage)  
 **If you want to use the trained model in another Python file:**
```bash
import torch
from src.model import SkinCancerModelV2


# 1. Initialize the model
model = SkinCancerModelV2()

# 2. Load the latest weights (file saved after training)
# model.load_state_dict(torch.load("models/dermatolog_v3_1.pth"))

model.eval()
print("Model loaded successfully and ready for testing!")
```
## 🧠 Multimodal Fusion (Hybrid Diagnosis) - NEW!
```bash
python src/hybrid_predict.py
# Output Example:
# 📸 Image Risk : %95.38
# ✍️ Complaint Risk : %99.92
# 🧠 HYBRID SCORE : %98.10
# 🩺 DIAGNOSIS : ⚠️ RISKY (Consult a Specialist)
```

## NLP Inference (Symptom Analysis)
```bash
from src.predict import predict_symptom
# Analyze text: "My color has darkened" -> Output: %96 Risky.
```
