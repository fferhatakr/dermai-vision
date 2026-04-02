
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold 
import sys

sys.path.append(os.getcwd())
from src.training.trainer_core import DermatologLightning
from src.dataloader.image_dataset import val_album , AlbumentationsDataset


DATA_DIR = "data/processed/full_dataset"
CSV_PATH = "data/processed/full_metadata.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_FOLDS = 5


def main():
    df = pd.read_csv(CSV_PATH)
    df['image'] = df['image'].str.replace('_downsampled', '')
    df['lesion_id'] = df['lesion_id'].fillna(df['image'])
    df.set_index('image', inplace=True)

    dataset = AlbumentationsDataset(DATA_DIR, album_transform=val_album)
    clean_file_names = []
    for f in dataset.imgs:
        raw_name = os.path.splitext(os.path.basename(f[0]))[0]
        clean_name = raw_name.replace('_downsampled', '')
        clean_file_names.append(clean_name)

    valid_imgfolder_indices = []
    valid_targets = []
    valid_groups = []
    valid_image_ids = []

    for img_idx, name in enumerate(clean_file_names):
        if name in df.index:
            valid_imgfolder_indices.append(img_idx)
            valid_targets.append(df.loc[name, 'targets'])
            valid_groups.append(df.loc[name, 'lesion_id'])
            valid_image_ids.append(name)

    valid_imgfolder_indices = np.array(valid_imgfolder_indices)
    valid_targets = np.array(valid_targets)
    valid_groups = np.array(valid_groups)
    valid_image_ids = np.array(valid_image_ids)

    sgkf = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    all_preds = []
    all_targets = []
    all_image_ids_ordered = []

    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(np.zeros(len(valid_targets)), valid_targets, groups=valid_groups)):
        
        model_name = "en_iyi_colab.ckpt"
        model_path = os.path.join("models", "ColabNew", model_name)
        
        if not os.path.exists(model_path):
            print(f"Skipping Fold {fold+1}: Model checkpoint not found at {model_path}")
            continue

        print(f"\nProcessing Fold {fold+1}")
        model = DermatologLightning.load_from_checkpoint(model_path, strict=False)
        model.to(DEVICE)
        model.eval()

        real_val_indices = valid_imgfolder_indices[val_idx]
        fold_image_ids = valid_image_ids[val_idx]

        val_subset = Subset(dataset, real_val_indices.tolist())
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

        all_image_ids_ordered.extend(fold_image_ids)

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Extracting Probabilities (Fold {fold+1})"):
                images = images.to(DEVICE)
                logits = model(images)
                pred = F.softmax(logits, dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(labels.numpy())
        break
    if len(all_preds) == 0:
        print("Error: No predictions generated. Ensure your trained .ckpt models are in the correct directory.")
        return

    class_names = ['0_mel', '1_nv', '2_bcc', '3_ak', '4_bkl', '5_df', '6_vasc', '7_scc']
    df_preds = pd.DataFrame(all_preds, columns=class_names)
    df_preds['image'] = all_image_ids_ordered
    df_preds['targets'] = all_targets

    big_df = pd.read_csv(CSV_PATH)
    df_final = pd.merge(df_preds, big_df, on='image')


    mean_age = df_final['age_approx'].median()
    df_final['age_approx'] = df_final['age_approx'].fillna(mean_age)
    df_final['sex'] = df_final['sex'].fillna('unknown')
    df_final['anatom_site_general'] = df_final['anatom_site_general'].fillna('unknown')
    
    output_path = "data/processed/oof_meta_dataset.csv"
    df_final.to_csv(output_path, index=False)
    print(f"\nSUCCESS: OOF Meta-Dataset created at {output_path}")

if __name__ == "__main__":
    main()

