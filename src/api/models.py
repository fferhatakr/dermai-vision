import torch
import joblib
import xgboost as xgb
import os
import sys

sys.path.append(os.getcwd())
from src.training.trainer_core import DermatologLightning


CKPT_PATH = "models/kfold_models/ultimate_v5_fold_4.ckpt"
XGB_MODEL_PATH = "models/xgb_meta_learner.json"
XGB_FEATURES_PATH = "models/xgb_features.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['0_mel', '1_nv', '2_bcc', '3_ak', '4_bkl', '5_df', '6_vasc', '7_scc']


lightning_model = None
xgb_model = None
feature_columns = None

def load_ai_models():
    global lightning_model, xgb_model, feature_columns

    lightning_model = DermatologLightning.load_from_checkpoint(
        CKPT_PATH, 
        strict=False
        )
    
    lightning_model.to(DEVICE).eval()
    
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(XGB_MODEL_PATH)
    feature_columns = joblib.load(XGB_FEATURES_PATH)

    return lightning_model ,xgb_model,feature_columns

