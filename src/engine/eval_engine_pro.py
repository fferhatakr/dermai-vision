import torch
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import torch.nn.functional as F
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())
from src.training.trainer_core import DermatologLightning 

CKPT_PATH = "models/ultimate_isic_PRO_v4.ckpt" 
TEST_DATA_DIR = "Data/processed/val" 
CLASS_NAMES = ['0_mel', '1_nv', '2_bcc', '3_ak', '4_bkl', '5_df', '6_vasc', '7_scc']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    print(f" Loading model on {DEVICE}: {CKPT_PATH}")
    
    model = DermatologLightning.load_from_checkpoint(CKPT_PATH, class_weights=torch.ones(8), strict=False)
    model.to(DEVICE)
    model.eval()
    return model

def predict_with_tta(model, img_path,num_views=5):
    img = Image.open(img_path).convert("RGB")

    base_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])
    ])


    tta_transform = transforms.Compose([
            transforms.Resize((300,300)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0,translate=(0.1,0.1),scale=(0.9,1.1)),
            transforms.ColorJitter(brightness=0.1,contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_list = [base_transform(img)]
    for _ in range(num_views - 1):
        img_list.append(tta_transform(img))
    
    batch_images = torch.stack(img_list).to(DEVICE)
    
    with torch.no_grad():
       
        logits = model(batch_images)
        probs = F.softmax(logits, dim=1)
        avg_probs = probs.mean(dim=0, keepdim=True)
        mel_prob = avg_probs[0][0].item() 
        threshold = 0.18
        
        if mel_prob > threshold:
            return 0 
        
        return torch.argmax(avg_probs, dim=1).item()

def run_test():
    print("TEST STARTING...")
    model = load_model()
    
    y_true = []
    y_pred = []

    
    for class_idx, class_name in enumerate(os.listdir(TEST_DATA_DIR)):
        class_path = os.path.join(TEST_DATA_DIR, class_name)
        if not os.path.isdir(class_path): continue

        try:
            actual_label = int(class_name.split('_')[0])
        except ValueError:
            print(f"ERROR: Folder name format is incorrect: {class_name}. It should be in the '0_mel' format")
            continue
        
        print(f" Testing {class_name}...")
        for img_file in tqdm(os.listdir(class_path)):
            img_path = os.path.join(class_path, img_file)
            
            pred_label = predict_with_tta(model, img_path)
            
            y_true.append(actual_label)
            y_pred.append(pred_label)

    print("\n" + "="*60)
    print("Detail")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    print("="*60)

    print("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, 
                yticklabels=CLASS_NAMES)
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Dermatolog AI - Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    run_test()


