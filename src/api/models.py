import torch
import joblib
import xgboost as xgb
import os
import sys
import onnxruntime as ort 

sys.path.append(os.getcwd())
from src.engine.trainer_core import DermatologLightning
from ultralytics import YOLO

YOLO_PATH = "experiments/models/detector/best.pt"
CKPT_PATH = "experiments/models/vision/midas_model.ckpt"  
ONNX_PATH = "experiments/models/onnx_model/midas_onnx"
XGB_MODEL_PATH = "experiments/models/meta/xgb_meta_learner.json"
XGB_FEATURES_PATH = "experiments/models/meta/xgb_features.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['0_mel', '1_nv', '2_bcc', '3_ak', '4_bkl', '5_df', '6_vasc', '7_scc']


lightning_model = None
xgb_model = None
feature_columns = None
yolo_model = None
onnx_session = None

def load_ai_models():
    global lightning_model, xgb_model, feature_columns ,yolo_model , onnx_session

    lightning_model = DermatologLightning.load_from_checkpoint(
        CKPT_PATH, 
        backbone="efficientnet_b3",
        strict=False
        )
    lightning_model.to(DEVICE).eval()

    onnx_session = ort.InferenceSession(
        ONNX_PATH,
        providers=["CPUExecutionProvider"]
    )

    
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(XGB_MODEL_PATH)
    feature_columns = joblib.load(XGB_FEATURES_PATH)

    
    yolo_model = YOLO(YOLO_PATH)

    return lightning_model ,xgb_model,feature_columns,yolo_model

