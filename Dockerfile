FROM python:3.10-slim

WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kodları kopyala
COPY src /app/src

# GÖRSELİNDEKİ GERÇEK YOLLAR VE İSİMLER:

# 1. Vision Modeli (Görselinde v2 yok, sadece best_model.ckpt)
COPY models/vision/best_model.ckpt /app/models/vision/

# 2. Meta Modelleri ve Encoderlar (.pk değil .pkl olmalı!)
COPY models/meta/xgb_meta_learner.json /app/models/meta/
COPY models/meta/xgb_features.pkl /app/models/meta/
COPY models/meta/le_sex.pkl /app/models/meta/
COPY models/meta/le_site.pkl /app/models/meta/

# 3. Detector (YOLO)
COPY models/detector/best.pt /app/models/detector/

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]