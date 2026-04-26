import torch
import joblib
import xgboost as xgb
import os
import sys
import onnxruntime as ort 
from hydra import initialize, compose

sys.path.append(os.getcwd())
from src.engine.trainer_core import DermatologLightning
from ultralytics import YOLO


try:
    initialize(config_path="../../configs", version_base=None, job_name="models_init")
except ValueError:
    pass  

cfg = compose(config_name="config")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = cfg.model.classes

lightning_model = None
xgb_model = None
feature_columns = None
yolo_model = None
onnx_session = None
le_sex = None
le_site = None

def load_ai_models():
    global lightning_model, xgb_model, feature_columns, yolo_model, onnx_session, le_sex, le_site
    
    lightning_model = DermatologLightning.load_from_checkpoint(
        cfg.paths.checkpoint_dir, 
        backbone=cfg.model.backbone,
        num_classes=len(cfg.model.classes),
        weight_decay=1e-4, 
        focal_gamma=2.0,    
        warmup_epochs=3,   
        strict=False
    )
    lightning_model.to(DEVICE).eval()

    onnx_session = ort.InferenceSession(
        cfg.paths.onnx_path,
        providers=["CPUExecutionProvider"]
    )

    
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(cfg.paths.xgb_model_path)
    feature_columns = joblib.load(cfg.paths.xgb_feature_path)   

    
    yolo_model = YOLO(cfg.paths.yolo_path)
    le_sex = joblib.load(cfg.paths.le_sex_path)
    le_site = joblib.load(cfg.paths.le_site_path)

    return lightning_model ,xgb_model,feature_columns,yolo_model

