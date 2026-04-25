
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
import hydra
from omegaconf import DictConfig

sys.path.append(os.getcwd())
from src.engine.trainer_core import DermatologLightning
from src.dataloader.image_dataset import val_album , AlbumentationsDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    data_dir = cfg.paths.data_path
    csv_path = cfg.meta.meta_csv_path
    k_folds = cfg.meta.meta_folds
    random_seed = cfg.train.random_seed
    model_path = cfg.meta.oof_model_path
    batch_size = cfg.train.batch_size
    num_workers = cfg.train.num_workers
    classes = cfg.meta.cnn_features
    output_path = cfg.meta.oof_output_path
    strip_suffix = cfg.meta.oof_strip_suffix
    image_col = cfg.meta.oof_image_col

    group_col = 'lesion_id'
    df = pd.read_csv(csv_path)
    df[image_col] = df[image_col].str.replace(strip_suffix, '')
    df[group_col] = df[group_col].fillna(df[image_col])
    df.set_index(image_col, inplace=True)

    dataset = AlbumentationsDataset(data_dir, album_transform=val_album)
    clean_file_names = []
    for f in dataset.imgs:
        raw_name = os.path.splitext(os.path.basename(f[0]))[0]
        clean_name = raw_name.replace(strip_suffix, '')
        clean_file_names.append(clean_name)

    valid_imgfolder_indices = []
    valid_targets = []
    valid_groups = []
    valid_image_ids = []

    for img_idx, name in enumerate(clean_file_names):
        if name in df.index:
            valid_imgfolder_indices.append(img_idx)
            valid_targets.append(df.loc[name, 'targets'])
            valid_groups.append(df.loc[name, group_col])
            valid_image_ids.append(name)

    valid_imgfolder_indices = np.array(valid_imgfolder_indices)
    valid_targets = np.array(valid_targets)
    valid_groups = np.array(valid_groups)
    valid_image_ids = np.array(valid_image_ids)

    sgkf = StratifiedGroupKFold(
        n_splits=k_folds, 
        shuffle=True, 
        random_state=random_seed
        )

    all_preds = []
    all_targets = []
    all_image_ids_ordered = []

    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(np.zeros(len(valid_targets)), valid_targets, groups=valid_groups)):

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
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        all_image_ids_ordered.extend(fold_image_ids)

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Extracting Probabilities (Fold {fold+1})"):
                images = images.to(DEVICE)
                logits = model(images)
                pred = F.softmax(logits, dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(labels.numpy())
    if len(all_preds) == 0:
        print("Error: No predictions generated. Ensure your trained .ckpt models are in the correct directory.")
        return
    
    df_preds = pd.DataFrame(all_preds, columns=classes)
    df_preds['image'] = all_image_ids_ordered
    df_preds['targets'] = all_targets

    big_df = pd.read_csv(csv_path)
    df_final = pd.merge(df_preds, big_df, on=image_col)


    mean_age = df_final['age_approx'].median()
    df_final['age_approx'] = df_final['age_approx'].fillna(mean_age)
    df_final['sex'] = df_final['sex'].fillna('unknown')
    df_final['anatom_site_general'] = df_final['anatom_site_general'].fillna('unknown')
    df_final.to_csv(output_path, index=False)
    print(f"\nSUCCESS: OOF Meta-Dataset created at {output_path}")

if __name__ == "__main__":
    main()

