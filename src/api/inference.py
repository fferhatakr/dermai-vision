import torch
import numpy as np
import cv2
import joblib
import xgboost as xgb
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import hydra
from omegaconf import DictConfig
import src.api.models as ai_models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    print(f"Inference module ready with {cfg.model.backbone}.")

def encode_clinical_data(age, sex, site):
    try:
        sex_encoded = ai_models.le_sex.transform([str(sex).lower()])[0]
    except:
        sex_encoded = ai_models.le_sex.transform(['unknown'])[0]

    try:
        site_encoded = ai_models.le_site.transform([str(site).lower()])[0]
    except:
        site_encoded = ai_models.le_site.transform(['unknown'])[0]
    
    return np.array([float(age), float(sex_encoded), float(site_encoded)])

def run_hybrid_inference(image_pil, age, sex, site, cfg: DictConfig, vision_model):
    vision_model.eval()
    
    tta_batch = apply_tta(image_pil, cfg) 
    with torch.no_grad():
        device = next(vision_model.parameters()).device
        tta_batch = tta_batch.to(device)
        logits = vision_model(tta_batch)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        avg_cnn_probs = np.mean(probs, axis=0)
        
    clinical_data = encode_clinical_data(age, sex, site)
    final_feature_vector = np.concatenate([avg_cnn_probs, clinical_data]).reshape(1, -1)
    
    final_probs = ai_models.xgb_model.predict_proba(final_feature_vector)[0]
    final_class = int(np.argmax(final_probs))
    class_names = cfg.model.classes

    return {
        "prediction": class_names[final_class],
        "confidence": float(final_probs[final_class]),
        "all_probabilities": dict(zip(class_names, final_probs.tolist())),
        "cnn_contribution": dict(zip(class_names, avg_cnn_probs.tolist())) 
    }

def apply_tta(image, cfg:DictConfig):
    norm_mean = cfg.model.mean
    norm_std = cfg.model.std
    img_sz = cfg.model.image_size

    base_transform = transforms.Compose([
        transforms.Resize((img_sz,img_sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    tta_transform = transforms.Compose([
        transforms.Resize((img_sz,img_sz)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    images = [base_transform(image)]
    for _ in range(4): 
        images.append(tta_transform(image))
    return torch.stack(images).to("cpu")

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

def run_quick_scan(image_pil,cfg: DictConfig):
    img_sz = cfg.model.image_size

    transform = transforms.Compose([
        transforms.Resize((img_sz, img_sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.model.mean, std=cfg.model.std)
    ])

    img_tensor = transform(image_pil).unsqueeze(0).numpy().astype(np.float32)

    input_name = ai_models.onnx_session.get_inputs()[0].name
    logits = ai_models.onnx_session.run(["logits"], {input_name: img_tensor})[0]

    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[0]
    
    MALIGNANT_INDICES = [0, 2, 3, 7]
    top_class = int(np.argmax(probs))
    is_risky = top_class in MALIGNANT_INDICES

    confidence = float(probs[top_class])
    low_confidence = confidence < 0.35

    class_names = cfg.model.classes

    return {
        "mode": "quick_scan",
        "is_risky": is_risky,
        "top_class": class_names[top_class],
        "confidence": confidence,
        "low_confidence_warning": low_confidence,
        "message": "RISKY — Expert assessment recommendations" if is_risky else "Low Risk"
    }

def run_standard_analysis(image_pil, age, sex, site, cfg: DictConfig):
    img_sz = cfg.model.image_size
    transform = transforms.Compose([
        transforms.Resize((img_sz, img_sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.model.mean, std=cfg.model.std)
    ])

    img_tensor = transform(image_pil).unsqueeze(0).numpy().astype(np.float32)

    input_name = ai_models.onnx_session.get_inputs()[0].name
    logits = ai_models.onnx_session.run(["logits"], {input_name: img_tensor})[0]
    cnn_probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[0]

    clinical_data = encode_clinical_data(age, sex, site)
    feature_vector = np.concatenate([cnn_probs, clinical_data]).reshape(1, -1)

    final_probs = ai_models.xgb_model.predict_proba(feature_vector)[0]
    final_class = int(np.argmax(final_probs))

    class_names = cfg.model.classes

    confidence = float(final_probs[final_class])
    low_confidence = confidence < 0.35

    return {
        "mode": "standard_analysis",
        "prediction": class_names[final_class],
        "confidence": confidence,
        "low_confidence_warning": low_confidence,
        "all_probabilities": dict(zip(class_names, final_probs.tolist())),
        "cnn_contribution": dict(zip(class_names, cnn_probs.tolist()))
    }