# 🩺 Skin Cancer Detection - Cepteki Dermatolog (v3 - Gelişmiş Versiyon)

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Status](https://img.shields.io/badge/Status-Geliştirme_Aşamasında-green.svg)

Bu proje, PyTorch kullanılarak geliştirilmiş, derin öğrenme tabanlı bir cilt kanseri sınıflandırma asistanıdır. Proje, sadece düz katmanlı modellerden (Linear) başlayıp, **CNN (Convolutional Neural Networks)** mimarisine ve **Veri Çoğaltma (Data Augmentation)** tekniklerine kadar uzanan bir mühendislik yolculuğunu kapsamaktadır.

## 🚀 Modelin Evrimi ve Performans Tablosu

Proje aşama aşama geliştirilmiş ve her versiyonda modelin genelleyebilirliği artırılmıştır.

| Versiyon         | Mimari       | Teknik                | Test Doğruluğu | Ortalama Hata (Loss) |
| :--------------- | :----------- | :-------------------- | :------------- | :------------------- |
| **v1**           | Linear (MLP) | Baseline              | %68.83         | 0.9014               |
| **v2**           | CNN          | 3-Layer Convolutional | %69.25         | 0.7658               |
| **v3 (Current)** | CNN          | **Data Augmentation** | **%70.50**     | **0.7950**           |

> **Mühendislik Notu:** v3 versiyonunda başarı oranının v2'ye yakın olması, modelin resimleri ezberlemeyi (Overfitting) bırakıp, farklı açılardan gelen verileri gerçekten öğrenmeye başladığını (Generalization) kanıtlamaktadır.

## 📊 Gelişmiş Analiz: Karmaşıklık Matrisi (Confusion Matrix)

Modelin hangi hastalık sınıflarında zorlandığını ve hangi sınıflarda uzmanlaştığını analiz etmek için Confusion Matrix kullanılmıştır. v3 versiyonu ile birlikte modelin farklı ışık ve açılardaki başarısı artırılmıştır.

## 📂 Dosya Yapısı

```text
AI_DET_PROJECT/
├─ Data/
├─ models/
│  ├─ cepteki_dermatolog_linear_v1.pth
│  └─ dermatolog_v2_agirliklar.pth
├─ notebooks/
│  ├─ v1_dermatolog.ipynb (v1 Çalışmaları)
│  └─ v2_dermatolog.ipynb (Augmentation Deneyleri)
│
├─ src/
│  ├─ __init_py
│  ├─ dataset.py
│  ├─ model.py
│  ├─train.py
│  └─ utils.py
│
├─ requirements.txt
└─ README.md
```

## 🛠️ Kullanılan Teknolojiler ve Teknikler

- **Mimari: 3 Katmanlı CNN (Conv2d, ReLU, MaxPool2d)**
- **Regülarizasyon: Dropout (0.5)**
- **Data Augmentation: RandomHorizontalFlip, RandomRotation (20°), ColorJitter**
- **Optimizasyon: Adam Optimizer (LR: 0.0001)**
- **Loss Function: CrossEntropyLoss**

## 🎯 Yol Haritası (Roadmap)
* **[x] v2: CNN mimarisine geçiş.**
* **[x] v2.1: Data Augmentation ile modelin güvenilirliğini artırma.**
* **[ ] v3.1: Class Weights (Sınıf Ağırlıkları) ile dengesiz veri sorununun çözümü.**
* **[ ] v4: Mobile Deployment (PyTorch Mobile ile Android entegrasyonu).**

## ⚙️ Kurulum

1. Repoyu klonlayın:

```bash
git clone https://github.com/kullanici_adiniz/AI_DET_PROJECT.git
cd AI_DET_PROJECT
```

2. Sanal ortam oluşturma:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Gerekli paketleri yükleyin:

```bash
pip install -r requirements.txt
```

## Kullanım

```bash
from src.model import SkinCancerModel
import torch

model = SkinCancerModel()
model.load_state_dict(torch.load("models/cepteki_dermatolog_linear_v1.pth"))
model.eval()
```

## Notebook üzerinden model eğitimi ve testleri yapılabilir.  
**📊 Mevcut Performans**  
-Test doğruluğu: %68.83  
-Ortalama hata: 0.9014  

**🚀 Geliştirme**

-Daha büyük ve dengeli veri setleri ile eğitim  
-Veri augmentasyonu ekleme  
-Farklı mimariler deneme (ResNet, EfficientNet)  


**Geliştirici: Ferhat Akar - Bilgisayar Mühendisliği Öğrencisi @OMÜ**