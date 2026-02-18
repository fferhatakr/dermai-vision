# 🩺 Skin Cancer Detection - Dermatologist in Your Pocket (v3.1 - Enhanced Version)

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Status](https://img.shields.io/badge/Status-Under_Development-green.svg)

This project is a deep learning-based skin cancer classification assistant developed using PyTorch. The project covers an engineering journey that starts from flat-layer models (Linear) and extends to **CNN (Convolutional Neural Networks)** architecture, **Data Augmentation** techniques, and **Class Weights** integration for imbalanced datasets.

## 🚀 Model Evolution and Performance Table

The project was developed step by step, with each version improving the model's real-world data adaptation (Generalization).

| Version | Architecture | Technique | Test Accuracy | Average Loss | Key Improvement |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v1** | Linear (MLP) | Baseline | 68.83% | 0.9014 | Basic skeleton established. |
| **v2** | CNN | 3-Layer Conv | 69.25% | 0.7658 | (Internal Test) Switched to CNN but overfitting was observed. |
| **v3** | CNN | Data Augmentation | 70.50% | 0.7950 | Overfitting broken, overall accuracy increased. |
| **v3.1** | CNN | **Class Weights** | **49.58%** | **1.1857** | **Justice System (Miracle):** While overall accuracy appears to have dropped, the diagnostic blindness for rare diseases (Class 3 and Class 6) that were previously never detected (scoring 0) has been completely eliminated! |

> **Engineering Note (v3.1):** When working with imbalanced datasets in medical AI projects, "Accuracy" (Overall Success) is a misleading metric. In v3.1, the model's "Majority Class (Class 5)" memorization was penalized and broken using Class Weights, forcing the system to search for rare/risky diseases. As a result, Accuracy dropped to the 49% range, but the model's **life-saving potential (catching rare cases)** has peaked. For more detailed analysis, the Confusion Matrix outputs can be examined.

## 📊 Advanced Analysis: Confusion Matrix

A Confusion Matrix was used to analyze which disease classes the model struggles with and which ones it specializes in. With version v3.1, the system has gained the ability to detect risky and rare diseases.

## 📂 File Structure

```text
AI_DET_PROJECT/
├─ Data/
├─ models/
│  ├─ cepteki_dermatolog_linear_v1.pth
│  └─ dermatolog_v2_agirliklar.pth
├─ notebooks/
│  ├─ v1_dermatolog.ipynb (v1 Work)
│  ├─ v2_dermatolog.ipynb (Augmentation Experiments)
│  └─ v3_dermatolog.ipynb (Class Weights Analysis)
├─ src/
│  ├─ __init__.py
│  ├─ dataset.py
│  ├─ model.py
│  ├─ train.py
│  └─ utils.py
├─ requirements.txt
└─ README.md
```

## 🛠️ Technologies and Techniques Used

- **Architecture: 3-Layer CNN (Conv2d, ReLU, MaxPool2d)**
- **Regularization: Dropout (0.5)**
- **Data Augmentation: RandomHorizontalFlip, RandomRotation (20°), ColorJitter**
- **Imbalanced Data Solution: Class Weights (sklearn.utils.class_weight)**  
- **Optimization: Adam Optimizer (LR: 0.0001)**
- **Loss Function: CrossEntropyLoss**

## 🎯 Roadmap
* **[x] v2: Migration to CNN architecture.**
* **[x] v2.1: Improving model reliability with Data Augmentation.**
* **[x] v3.1: Solving the imbalanced data problem with Class Weights.**
* **[ ] v4: Maximizing accuracy with Transfer Learning (ResNet, EfficientNet).**  
* **[ ] v5: Mobile Deployment (Android integration with PyTorch Mobile).**

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
# To start training and testing (v3.1 Architecture)
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
