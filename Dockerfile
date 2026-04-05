FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src
COPY configs /app/configs

# Vision model
COPY experiments/models/vision/midas_model.ckpt /app/experiments/models/vision/

# ONNX model
COPY experiments/models/onnx_model/midas_onnx /app/experiments/models/onnx_model/midas_onnx

# Meta learner
COPY experiments/models/meta/xgb_meta_learner.json /app/experiments/models/meta/
COPY experiments/models/meta/xgb_features.pkl /app/experiments/models/meta/
COPY experiments/models/meta/le_sex.pkl /app/experiments/models/meta/
COPY experiments/models/meta/le_site.pkl /app/experiments/models/meta/

# YOLO detector
COPY experiments/models/detector/best.pt /app/experiments/models/detector/

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]