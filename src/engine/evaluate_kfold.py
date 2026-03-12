import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.training.trainer_core import DermatologLightning
from src.dataloader.image_dataset import  val_transforms




TEST_DATA_DIR = "Data/processed/kfold_data" 
CLASS_NAMES = ['0_mel', '1_nv', '2_bcc', '3_ak', '4_bkl', '5_df', '6_vasc', '7_scc']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
dataset = datasets.ImageFolder(TEST_DATA_DIR, transform=val_transforms)
targets = np.array(dataset.targets)

all_preds = []
all_true = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(targets)), targets)):
    

    model_path = f"models/kfold_models/ultimate_v5_fold_{fold+1}.ckpt"
    model = DermatologLightning.load_from_checkpoint(model_path,strict=False)
    model.to(DEVICE)
    model.eval()

    val_sub = Subset(dataset, val_idx)
    val_loader = DataLoader(val_sub, batch_size=16, shuffle=False)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Evaluating Fold {fold+1}")
        for images , labels in pbar:
            images = images.to(DEVICE)

            logits = model(images)
            preds = torch.argmax(logits,dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.numpy())
    print(f"Fold {fold+1} predictions completed.")

print(classification_report(all_true, all_preds, target_names=dataset.classes))
cm = confusion_matrix(all_true,all_preds)
plt.figure(figsize=(12, 10)) 
sns.heatmap(
    cm, 
    annot=True,               
    fmt='d',                 
    cmap='Blues',             
    xticklabels=dataset.classes, 
    yticklabels=dataset.classes
)

plt.title('Final Confusion Matrix (5-Fold Merged)')
plt.ylabel('True')
plt.xlabel('Predicted')

plt.savefig("confusion_matrix_final.png") 
plt.show()
