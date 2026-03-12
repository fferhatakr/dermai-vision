import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import sys

sys.path.append(os.getcwd())
from src.training.trainer_core import DermatologLightning
from src.dataloader.image_dataset import val_transforms


DATA_DIR = "data/processed/kfold_data"
CSV_PATH = "data/raw/ISIC_2019_Training_Metadata.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main():
    dataset = datasets.ImageFolder(DATA_DIR,transform=val_transforms)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    targets = np.array(dataset.targets)

    all_preds = []
    all_targets = []
    all_image_ids = []


    for fold ,(train_idx,val_idx) in enumerate(kfold.split(np.zeros(len(targets)), targets)):
        model_path = f"models/kfold_models/ultimate_v5_fold_{fold+1}.ckpt"
        model = DermatologLightning.load_from_checkpoint(model_path,strict=False)
        model.to(DEVICE)
        model.eval()


        val_subset = Subset(dataset,val_idx)
        val_loader = DataLoader(
            val_subset,
            batch_size=32,
            shuffle=False,
            num_workers=8
        ) 


        for i in val_idx:
            name = os.path.basename(dataset.samples[i][0]).split('.')[0]
            all_image_ids.append(name)

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Fold {fold+1}"):
                images = images.to(DEVICE)
                logits = model(images)
                pred = F.softmax(logits,dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(labels.numpy())

    class_names = ['0_mel', '1_nv', '2_bcc', '3_ak', '4_bkl', '5_df', '6_vasc', '7_scc']
    df = pd.DataFrame(all_preds, columns=class_names)
    df['image'] = all_image_ids
    df['targets'] = all_targets

    big_df = pd.read_csv(CSV_PATH)
    df_final = pd.merge(df, big_df, on='image')

    mean_age = df_final['age_approx'].median()
    df_final['age_approx'] = df_final['age_approx'].fillna(mean_age)
    df_final['sex'] = df_final['sex'].fillna('unknown')
    df_final['anatom_site_general'] = df_final['anatom_site_general'].fillna('unknown')
    
    
    df_final.to_csv("data/processed/oof_meta_dataset.csv", index=False)
    print("OOF Meta-Dataset Successfully Created.")

if __name__ == "__main__":
    main()