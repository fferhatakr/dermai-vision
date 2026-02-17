# 🩺 Skin Cancer Detection - Cepteki Dermatolog (v3.1 - Gelişmiş Versiyon)

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Status](https://img.shields.io/badge/Status-Geliştirme_Aşamasında-green.svg)

Bu proje, PyTorch kullanılarak geliştirilmiş, derin öğrenme tabanlı bir cilt kanseri sınıflandırma asistanıdır. Proje, sadece düz katmanlı modellerden (Linear) başlayıp, **CNN (Convolutional Neural Networks)** mimarisine, **Veri Çoğaltma (Data Augmentation)** tekniklerine ve dengesiz veri setleri için **Class Weights (Sınıf Ağırlıkları)** entegrasyonuna kadar uzanan bir mühendislik yolculuğunu kapsamaktadır.

## 🚀 Modelin Evrimi ve Performans Tablosu

Proje aşama aşama geliştirilmiş ve her versiyonda modelin gerçek dünya verilerine uyumu (Generalization) artırılmıştır.

| Versiyon | Mimari | Teknik | Test Doğruluğu | Ortalama Hata (Loss) | Önemli Gelişme |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v1** | Linear (MLP) | Baseline | %68.83 | 0.9014 | Temel iskelet kuruldu. |
| **v2** | CNN | 3-Layer Conv | %69.25 | 0.7658 | (Internal Test) CNN'e geçildi ama ezber sorunu görüldü. |
| **v3** | CNN | Data Augmentation | %70.50 | 0.7950 | Ezber bozuldu, genel doğruluk arttı. |
| **v3.1** | CNN | **Class Weights** | **%49.58** | **1.1857** | **Adalet Sistemi (Mucize):** Genel başarı düşmüş gibi görünse de, daha önce hiç tespit edilemeyen (0 çeken) nadir hastalıklardaki (Sınıf 3 ve Sınıf 6) teşhis körlüğü tamamen ortadan kaldırıldı! |

> **Mühendislik Notu (v3.1):** Tıbbi yapay zeka projelerinde dengesiz veri setleriyle çalışırken "Accuracy" (Genel Başarı) yanıltıcı bir metriktir. v3.1'de modelin "Çoğunluk Sınıfı (Sınıf 5)" ezberi Class Weights ile cezalandırılmış ve bozularak, sistem nadir/riskli hastalıkları aramaya zorlanmıştır. Bu nedenle Accuracy %49 bandına inmiş ancak modelin **hayat kurtarma potansiyeli (nadir vakaları yakalama)** zirveye çıkmıştır. Daha detaylı analiz için Karmaşıklık Matrisi (Confusion Matrix) çıktıları incelenebilir.

## 📊 Gelişmiş Analiz: Karmaşıklık Matrisi (Confusion Matrix)

Modelin hangi hastalık sınıflarında zorlandığını ve hangi sınıflarda uzmanlaştığını analiz etmek için Confusion Matrix kullanılmıştır. v3.1 versiyonu ile birlikte sistem, riskli ve nadir hastalıkları tespit etme yeteneği kazanmıştır.

## 📂 Dosya Yapısı

```text
AI_DET_PROJECT/
├─ Data/
├─ models/
│  ├─ cepteki_dermatolog_linear_v1.pth
│  └─ dermatolog_v2_agirliklar.pth
├─ notebooks/
│  ├─ v1_dermatolog.ipynb (v1 Çalışmaları)
│  ├─ v2_dermatolog.ipynb (Augmentation Deneyleri)
│  └─ v3_dermatolog.ipynb (Class Weights Analizi)
├─ src/
│  ├─ __init__.py
│  ├─ dataset.py
│  ├─ model.py
│  ├─ train.py
│  └─ utils.py
├─ requirements.txt
└─ README.md
```

## 🛠️ Kullanılan Teknolojiler ve Teknikler

- **Mimari: 3 Katmanlı CNN (Conv2d, ReLU, MaxPool2d)**
- **Regülarizasyon: Dropout (0.5)**
- **Data Augmentation: RandomHorizontalFlip, RandomRotation (20°), ColorJitter**
- **Dengesiz Veri Çözümü: Class Weights (sklearn.utils.class_weight)**  
- **Optimizasyon: Adam Optimizer (LR: 0.0001)**
- **Loss Function: CrossEntropyLoss**

## 🎯 Yol Haritası (Roadmap)
* **[x] v2: CNN mimarisine geçiş.**
* **[x] v2.1: Data Augmentation ile modelin güvenilirliğini artırma.**
* **[x] v3.1: Class Weights (Sınıf Ağırlıkları) ile dengesiz veri sorununun çözümü.**
* **[ ] v4: Transfer Learning (ResNet, EfficientNet) ile başarı oranını maksimize etme.**  
* **[ ] v5: Mobile Deployment (PyTorch Mobile ile Android entegrasyonu).**

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

### Notebook üzerinden model eğitimi ve testleri yapılabilir.  
## 🚀 Geliştirme

- **Daha büyük ve dengeli veri setleri ile eğitim** 
- **Veri augmentasyonu ekleme**  
- **Farklı mimariler deneme (ResNet, EfficientNet)**    


### Geliştirici: Ferhat Akar - Bilgisayar Mühendisliği Öğrencisi @OMÜ