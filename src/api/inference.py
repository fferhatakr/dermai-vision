import torch
import numpy as np
import cv2
import joblib
import xgboost as xgb
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from src.api.models import DEVICE

# Modelleri ve Encoder'ları Yükle (Path'leri temizlik sonrası yerlerine göre güncelledim)
XGB_MODEL_PATH = "models/meta/xgb_meta_learner.json"
XGB_FEATURES_PATH = "models/meta/xgb_features.pkl"
LE_SEX_PATH = "models/meta/le_sex.pkl"
LE_SITE_PATH = "models/meta/le_site.pkl"

# Global olarak yükleyelim ki her istekte disk okunmasın
meta_learner = xgb.XGBClassifier()
meta_learner.load_model(XGB_MODEL_PATH)
feature_names = joblib.load(XGB_FEATURES_PATH)
le_sex = joblib.load(LE_SEX_PATH)
le_site = joblib.load(LE_SITE_PATH)

def run_hybrid_inference(image_pil, age, sex, site, vision_model):
    """
    CNN (EfficientNet) + Metadata (XGBoost) hibrit tahmin mekanizması.
    """
    vision_model.eval()
    
    # 1. Image Preprocessing & TTA
    # Opsiyonel: Vignette eklemek istersen burada çağırabilirsin
    # image_pil = apply_vignette(image_pil) 
    tta_batch = apply_tta(image_pil) # [5, 3, 300, 300] tensor döner
    
    # 2. Vision (CNN) Olasılıklarını Çıkar
    with torch.no_grad():
        logits = vision_model(tta_batch)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        # 5 TTA görüntüsünün ortalamasını alarak daha stabil bir sonuç elde edelim
        avg_cnn_probs = np.mean(probs, axis=0) # [8] boyutunda vektör
    
    # 3. Klinik Verileri Encode Et (XGBoost'un anladığı dile çevir)
    try:
        sex_encoded = le_sex.transform([str(sex).lower()])[0]
    except:
        sex_encoded = le_sex.transform(['unknown'])[0]
        
    try:
        site_encoded = le_site.transform([str(site).lower()])[0]
    except:
        site_encoded = le_site.transform(['unknown'])[0]

    # 4. 11 Boyutlu Feature Vector Hazırla
    # Sıralama: [0_mel, 1_nv, ..., 7_scc, age_approx, sex_encoded, site_encoded]
    clinical_data = np.array([age, sex_encoded, site_encoded])
    final_feature_vector = np.concatenate([avg_cnn_probs, clinical_data]).reshape(1, -1)
    
    # 5. Meta-Learner (XGBoost) Nihai Karar
    final_probs = meta_learner.predict_proba(final_feature_vector)[0]
    final_class = np.argmax(final_probs)
    
    class_names = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
    
    return {
        "prediction": class_names[final_class],
        "confidence": float(final_probs[final_class]),
        "all_probabilities": dict(zip(class_names, final_probs.tolist())),
        "cnn_contribution": dict(zip(class_names, avg_cnn_probs.tolist())) # CNN ne demişti?
    }

# Mevcut fonksiyonlarını koru (TTA ve Vignette)
def apply_tta(image):
    # ... (Gonderdiğin kodun aynısı buraya gelecek) ...
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
    # ... (Gonderdiğin kodun aynısı buraya gelecek) ...
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