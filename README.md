# 🩺 Skin Cancer Detection - Dermatologist in Your Pocket (v4.2 - Mobile Optimization)

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Status](https://img.shields.io/badge/Status-Under_Development-green.svg)

This project is a deep learning-based skin cancer classification assistant developed using PyTorch. The project covers an engineering journey that starts from flat-layer models (Linear), extends to custom CNN architectures, and currently utilizes **Transfer Learning (ResNet18 & MobileNetV3)** for mobile-optimized, high-accuracy predictions.

### 🚀 Model Evolution and Performance Table

The project was developed step by step, with each version improving the model's real-world data adaptation and deployment readiness.

## Vision Models (Image Analysis)
| Version | Architecture | Technique | Test Accuracy | Average Loss | Key Improvement |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v1** | Linear (MLP) | Baseline | 68.83% | 0.9014 | Basic skeleton established. |
| **v3.1** | Custom CNN | Class Weights | 49.58% | 1.1857 | Justice System (Miracle): Overfitting broken, diagnostic blindness for rare diseases eliminated. |
| **v4.0** | **ResNet18** | **Full Retraining** | **78.75%** | **0.7465** | **Transfer Learning Revolution:** Integrated pre-trained ImageNet weights, massive jump in understanding skin lesion features. |
| **v4.2** | **MobileNetV3-Small** | **Mobile Optimization** | **73.67%** | **0.8120** | **On-Device Ready:** Swapped heavy ResNet for an ultra-lightweight mobile architecture. Minimal accuracy drop for massive battery/performance gain on mobile devices. |

> **Engineering Note (v4.2):** Hitting ~74% accuracy with a lightweight model like MobileNetV3-Small on a highly imbalanced, 7-class medical dataset is a massive optimization milestone. The model is now perfectly sized to be converted into TorchScript for native Android (Kotlin) deployment without draining device resources.


## NLP Models (Symptom Analysis) - NEW!
| Version | Architecture | Dataset | Accuracy | Key Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **v1.0** | DistilBERT (TR) | Custom Dataset | 96.08% | Semantic Understanding: Detecting risk factors in text. |


**Engineering Note (v5.0): The project is now in the "Multimodal" phase. It supports diagnostic accuracy by focusing not only on pixels but also on the patient's written complaints such as "rapid growth" and "bleeding".**  

## 📂 File Structure

```text
AI_DET_PROJECT/
├─ Data/
│  ├─ train/ (Image Dataset)
│  └─ symptoms.csv (NLP Training Data)
├─ models/
│  ├─ dermatolog_v4.2.pth (MobileNet Weights)
│  └─ nlp_v1/ (DistilBERT Model & Tokenizer)
├─ src/
│  ├─ dataset.py (Data Augmentation & Normalization)
│  ├─ model.py (Transfer Learning Architectures)
│  ├─ train.py (Dynamic LR & Full Retraining logic)
│  ├─ nlp_dataset.py
│  ├─ predict.py
│  ├─ nlp_model.py & nlp_dataset.py (NLP Pipeline)
│  ├─ nlp_train.py (NLP Training Script)
│  └─ utils.py
├─ requirements.txt
└─ README.md
```

## 🛠️ Technologies and Techniques Used

- **Architectures: Custom CNNs, ResNet18, MobileNetV3**
- **Transfer Learning: Fine-tuning pre-trained ImageNet weights (requires_grad=True, low learning rate)**  
- **Data Pipeline: RandomHorizontalFlip, RandomRotation, ColorJitter, ImageNet Normalization.** 
- **Imbalanced Data Solution: Class Weights (sklearn) for vision; data augmentation for NLP.** 
- **Optimization: AdamW optimizer, Dynamic Learning Rate, Softmax Probability Scoring.**  

## 🎯 Roadmap
* **[x] v2: Migration to CNN architecture.**
* **[x] v2.1: Improving model reliability with Data Augmentation.**
* **[x] v3.1: Solving the imbalanced data problem with Class Weights.**
* **[x] v4.0: Maximizing accuracy with Transfer Learning (ResNet18).**
* **[x] v4.2: Mobile optimization with MobileNetV3-Small.**
* **[x] v5.0: Multimodal NLP Integration (Symptom analysis)**
* **[ ] v5.1: Unified Multimodal Fusion (Combining Image + Text scores).**  
* **[ ] v7.0: Mobile Deployment (Android Kotlin integration).** 


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
python src/nlp_train.py

# 3. Train Vision Model
python src/train.py
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

## NLP Inference (Symptom Analysis)
```bash
from src.predict import predict_symptom
# Analyze text: "Benim rengi koyulaştı." -> Output: %96 Risky.
```
