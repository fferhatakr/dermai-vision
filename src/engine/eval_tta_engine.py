import torch
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import torch.nn.functional as F
import sys

sys.path.append(os.getcwd())
from src.training.trainer_core import DermatologLightning 

# --- CONFIGURATION ---
CKPT_PATH = "models/ultimate_model.ckpt" # Filename from training
TEST_DATA_DIR = "Data/external_test/final_clean_test"
CSV_PATH = "Data/external_test/test_data_new/metdadata_v2.csv" # Or your metadata_v2.csv path
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Diagnosis Mapping
MAPPING = {
    'actinic keratosis': 0, 'squamous cell carcinoma': 0,
    'basal cell carcinoma': 1,
    'seborrheic keratosis': 2, 'lichen planus-like keratosis': 2, 'solar lentigo': 2,
    'dermatofibroma': 3,
    'melanoma': 4,
    'nevus': 5,
    'vascular lesion': 6
}

def load_model():
    print(f"Loading model: {CKPT_PATH}")
    # Using strict=False to bypass weight mismatch
    model = DermatologLightning.load_from_checkpoint(CKPT_PATH, class_weights=None, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

# STRATEGY 4: TTA (Test Time Augmentation)
def predict_with_tta(model, img_path):
    img = Image.open(img_path).convert("RGB")
    
    # 1. Base Transform
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. TTA Variations
    # Original
    img_orig = base_transform(img).unsqueeze(0).to(DEVICE)
    # Horizontal Flip
    img_flip_h = base_transform(transforms.functional.hflip(img)).unsqueeze(0).to(DEVICE)
    # Vertical Flip
    img_flip_v = base_transform(transforms.functional.vflip(img)).unsqueeze(0).to(DEVICE)
    
    # 3. Batch Prediction
    with torch.no_grad():
        logits_orig = model(img_orig)
        logits_flip_h = model(img_flip_h)
        logits_flip_v = model(img_flip_v)
        
        # 4. Take Average (After applying Softmax)
        probs_orig = F.softmax(logits_orig, dim=1)
        probs_flip_h = F.softmax(logits_flip_h, dim=1)
        probs_flip_v = F.softmax(logits_flip_v, dim=1)
        
        avg_probs = (probs_orig + probs_flip_h + probs_flip_v) / 3.0
        
        pred_idx = torch.argmax(avg_probs, dim=1).item()
        
    return pred_idx

def run_test():
    print("TTA SUPPORTED FINAL TEST STARTING...")
    
    if not os.path.exists(CKPT_PATH):
        print("ERROR: Complete training (run_ultimate_training.py) first!")
        return

    model = load_model()
    metadata = pd.read_csv(CSV_PATH)
    id_col = 'image_name' if 'image_name' in metadata.columns else 'isic_id'
    metadata = metadata.set_index(id_col)

    y_true = []
    y_pred = []
    processed_count = 0

    test_files = [f for f in os.listdir(TEST_DATA_DIR) if f.lower().endswith('.jpg')]
    print(f"Scanning {len(test_files)} images with TTA (this will take a bit longer)...")

    for img_file in tqdm(test_files):
        img_id = os.path.splitext(img_file)[0]
        if img_id not in metadata.index: continue
        diagnosis = str(metadata.loc[img_id, 'diagnosis']).lower()
        if diagnosis not in MAPPING: continue
            
        true_label_idx = MAPPING[diagnosis]
        img_path = os.path.join(TEST_DATA_DIR, img_file)
        
        # Prediction with TTA
        pred_label_idx = predict_with_tta(model, img_path)
        
        y_true.append(true_label_idx)
        y_pred.append(pred_label_idx)
        processed_count += 1

    print("\n" + "="*60)
    print("ULTIMATE RESULT REPORT")
    
    # 1. Added labels here: "Set missing classes to 0" 
    print(classification_report(
        y_true, 
        y_pred, 
        target_names=CLASS_NAMES, 
        labels=range(len(CLASS_NAMES)), # <--- ADD THIS
        digits=4, 
        zero_division=0
    ))
    
    print("\nConfusion Matrix:")
    # 2. Added labels here to ensure matrix is 7x7
    print(confusion_matrix(
        y_true, 
        y_pred, 
        labels=range(len(CLASS_NAMES)) # <--- ADD THIS
    ))
    print("="*60)

if __name__ == "__main__":
    run_test()