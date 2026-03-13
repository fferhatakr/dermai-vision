FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY src /app/src

COPY models/kfold_models/ultimate_v5_fold_4.ckpt /app/models/kfold_models/
COPY models/xgb_meta_learner.json /app/models/
COPY models/xgb_features.pkl /app/models/

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]