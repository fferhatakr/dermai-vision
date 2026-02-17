# 🩺 Skin Cancer Detection - Cepteki Dermatolog (v1)

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)

Bu proje, PyTorch kullanılarak geliştirilmiş, derin öğrenme tabanlı bir cilt kanseri sınıflandırma modelidir. Projenin amacı, 7 farklı cilt lezyonu türünü analiz ederek erken teşhis sürecine yardımcı olmaktır.

## 📂 Dosya Yapısı

```text
AI_DET_PROJECT/
├─ Data/
├─ models/
│  └─ cepteki_dermatolog_linear_v1.pth
├─ notebooks/
│  └─ notebook.ipynb
├─ src/
│  ├─ __init__.py
│  ├─ dataset.py
│  ├─ model.py
│  ├─ train.py
│  └─ utils.py
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## 📊 Model Performansı (Baseline - Linear Model)

Şu anki sonuçlar, sadece **Düz (Linear)** katmanlar içeren ilk versiyona aittir.

| Metrik | Sonuç |
| :--- | :--- |
| **Test Doğruluğu** | %67.83 |
| **Ortalama Hata (Loss)** | 0.9303 |
| **Sınıf Sayısı** | 7 |



## 🚀 Yol Haritası (Roadmap)

- [ ] **v2:** CNN (Convolutional Neural Networks) mimarisine geçiş.
- [ ] **v2.1:** Data Augmentation (Veri Çeşitlendirme) ile modelin genelleyebilirliğini artırma.
- [ ] **v3:** Mobile Deployment (PyTorch Mobile ile Android entegrasyonu).

## 📂 Dosya Yapısı




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


**Notebook üzerinden model eğitimi ve testleri yapılabilir.**
📊 Mevcut Performans

Test doğruluğu: %68.83

Ortalama hata: 0.9014

🚀 Geliştirme

Daha büyük ve dengeli veri setleri ile eğitim

Veri augmentasyonu ekleme

Farklı mimariler deneme (ResNet, EfficientNet)