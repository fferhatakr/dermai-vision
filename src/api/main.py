from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
from PIL import Image
import io
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import pandas as pd
import cv2
import base64
import os
import sys
import joblib
import xgboost as xgb

sys.path.append(os.getcwd())
from src.training.trainer_core import DermatologLightning


CKPT_PATH = "models/kfold_models/ultimate_v5_fold_4.ckpt" 
XGB_MODEL_PATH = "models/xgb_meta_learner.json"
XGB_FEATURES_PATH = "models/xgb_features.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['0_mel', '1_nv', '2_bcc', '3_ak', '4_bkl', '5_df', '6_vasc', '7_scc']

app = FastAPI(title="DermaScan AI - Full Debug Meta-Engine")

lightning_model = None
xgb_model = None
feature_columns = None

@app.on_event("startup")
def load_ai_models():
    global lightning_model, xgb_model, feature_columns
    lightning_model = DermatologLightning.load_from_checkpoint(CKPT_PATH, strict=False)
    lightning_model.to(DEVICE).eval()
    
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(XGB_MODEL_PATH)
    feature_columns = joblib.load(XGB_FEATURES_PATH)
    

def apply_tta(image):
    base_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tta_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    images = [base_transform(image)]
    for _ in range(4): 
        images.append(tta_transform(image))
    return torch.stack(images).to(DEVICE)

def apply_vignette(image_pil, sigma=180):
    img_cv = np.array(image_pil)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    rows, cols = img_cv.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, sigma)
    kernel_y = cv2.getGaussianKernel(rows, sigma)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    mask = mask.astype(np.float32) / mask.max()
    mask_3ch = np.dstack([mask] * 3)
    vignette_img = (img_cv * mask_3ch).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(vignette_img, cv2.COLOR_BGR2RGB))

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...), 
    age: float = Form(...), 
    sex: str = Form(...), 
    anatom_site: str = Form(...),
    needs_heatmap: bool = Form(False)
):
    
    image_bytes = await file.read()
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    processed_image = apply_vignette(original_image)
    
    
    input_batch = apply_tta(processed_image)
    with torch.no_grad():
        logits = lightning_model(input_batch)
        cnn_probs = F.softmax(logits, dim=1).mean(dim=0).cpu().numpy()

    
    meta_data = {'age_approx': age, 'sex': sex, 'anatom_site_general': anatom_site}
    for i, class_name in enumerate(CLASSES):
        meta_data[class_name] = cnn_probs[i]

    df_meta = pd.DataFrame([meta_data])
    df_meta = pd.get_dummies(df_meta, columns=['sex', 'anatom_site_general'])
    for col in feature_columns:
        if col not in df_meta.columns: df_meta[col] = 0
    df_meta = df_meta[feature_columns]

    final_probs = xgb_model.predict_proba(df_meta)[0]
    top_idx = np.argmax(final_probs)
    cnn_top_idx = np.argmax(cnn_probs)

    
    malignant_indices = [0, 2, 3, 7]
    benign_indices = [1, 4, 5, 6]
    prob_mel = final_probs[0]

    cnn_top_idx = int(np.argmax(cnn_probs))
    cnn_max_conf = float(cnn_probs[cnn_top_idx])

    
    final_diagnosis = CLASSES[top_idx].upper()
   

    if top_idx in malignant_indices:
        
        is_risky = True
    elif cnn_top_idx in malignant_indices and cnn_max_conf > 0.40:
        
        is_risky = True
        final_diagnosis = f"{CLASSES[cnn_top_idx].upper()} (VISUAL ALERT)"
    elif prob_mel > 0.45:
        
        is_risky = True
    else:
        is_risky = False

    
    encoded_heatmap = ""
    if needs_heatmap:
        try:
            single_input = transforms.Compose([
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(processed_image).unsqueeze(0).to(DEVICE)
            hm = lightning_model.generate_gradcam(single_input)
            if isinstance(hm, torch.Tensor): hm = hm.squeeze().detach().cpu().numpy()
            hm = cv2.resize(hm, (original_image.size[0], original_image.size[1]))
            hm_uint8 = np.uint8(255 * hm)
            hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
            _, buf = cv2.imencode(".png", hm_color)
            encoded_heatmap = base64.b64encode(buf).decode("utf-8")
        except: pass

    
    top_idx = int(np.argmax(final_probs))       
    cnn_top_idx = int(np.argmax(cnn_probs))     
    
    
    has_conflict = bool(top_idx != cnn_top_idx) 

    
    hybrid_all_probs = {CLASSES[i].upper(): float(final_probs[i]) for i in range(len(CLASSES))}
    cnn_all_probs = {CLASSES[i].upper(): float(cnn_probs[i]) for i in range(len(CLASSES))}

    return {
        "prediction": "Risky" if is_risky else "Benign",
        "diagnosis": CLASSES[top_idx].upper(),
        "confidence": float(final_probs[top_idx]),
        "all_probabilities": dict(sorted(hybrid_all_probs.items(), key=lambda x: x[1], reverse=True)),
        "heatmap_base64": encoded_heatmap,
        "metadata_used": {"age": age, "sex": sex, "site": anatom_site},
        "debug": {
            "cnn_diagnosis": CLASSES[cnn_top_idx].upper(),
            "cnn_confidence": float(cnn_probs[cnn_top_idx]),
            "cnn_all_probabilities": dict(sorted(cnn_all_probs.items(), key=lambda x: x[1], reverse=True)),
            "conflict": has_conflict  
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)