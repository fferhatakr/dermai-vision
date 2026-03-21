import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets
import sys
import os
import torch.nn.functional as F

sys.path.append(os.getcwd())

from src.training.trainer_core import DermatologLightning
from src.dataloader.image_dataset import train_transforms, val_transforms


class UltimateDermatolog(DermatologLightning):
    def compute_loss(self, logits, labels):
        ce_loss = F.cross_entropy(
            logits, labels,
            weight=self.class_weights,
            label_smoothing=0.1,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** 2.0) * ce_loss
        return focal_loss.mean()


def main():
    CSV_PATH = "data/processed/oof_meta_dataset.csv"
    DATA_PATH = "data/processed/just_leke"
    BATCH_SIZE = 16
    EPOCHS = 25
    K_FOLDS = 5
    BACKBONE = "efficientnet_b3"  
    LR = 1e-4
    EXPERIMENT_NAME = f"v7_{BACKBONE}"

    print("Reading Metadata")
    df = pd.read_csv(CSV_PATH)
    df['image'] = df['image'].str.replace('_downsampled', '')
    df['lesion_id'] = df['lesion_id'].fillna(df['image'])
    df.set_index('image', inplace=True)

    
    full_dataset = datasets.ImageFolder(DATA_PATH)

    clean_file_names = []
    for f in full_dataset.imgs:
        raw_name = os.path.splitext(os.path.basename(f[0]))[0]
        clean_name = raw_name.replace('_downsampled', '')
        clean_file_names.append(clean_name)

    valid_imgfolder_indices = []
    valid_targets = []
    valid_groups = []
    skipped = 0

    for img_idx, name in enumerate(clean_file_names):
        if name in df.index:
            valid_imgfolder_indices.append(img_idx)
            valid_targets.append(df.loc[name, 'targets'])
            valid_groups.append(df.loc[name, 'lesion_id'])
        else:
            skipped += 1

    if skipped > 0:
        print(f"Warning: {skipped} The file could not be found in the CSV and was skipped")

    valid_imgfolder_indices = np.array(valid_imgfolder_indices)
    valid_targets = np.array(valid_targets)
    valid_groups = np.array(valid_groups)

    print(f"Total matching images: {len(valid_targets)}")
    print(f"Class distribution: {dict(zip(*np.unique(valid_targets, return_counts=True)))}")

    sgkf = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(np.zeros(len(valid_targets)), valid_targets, groups=valid_groups)):
        if fold > 0:
            break

        
        train_g = set(valid_groups[train_idx])
        val_g = set(valid_groups[val_idx])
        overlap = train_g.intersection(val_g)

        print("-" * 40)
        if len(overlap) == 0:
            print("LEAK TEST SUCCESSFUL")
        else:
            print(f"Please note: {len(overlap)}There’s a leak")
        print("-" * 40)

        
        real_train_indices = valid_imgfolder_indices[train_idx]
        real_val_indices = valid_imgfolder_indices[val_idx]

        
        dataset_train = datasets.ImageFolder(DATA_PATH, transform=train_transforms)
        dataset_val = datasets.ImageFolder(DATA_PATH, transform=val_transforms)

        train_sub = Subset(dataset_train, real_train_indices.tolist())
        val_sub = Subset(dataset_val, real_val_indices.tolist())

        curr_train_targets = valid_targets[train_idx]
        curr_val_targets = valid_targets[val_idx]

        print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")
        print(f"Train distribution:: {dict(zip(*np.unique(curr_train_targets, return_counts=True)))}")
        print(f"Val distribution:   {dict(zip(*np.unique(curr_val_targets, return_counts=True)))}")

        
        sample_checks = min(20, len(real_train_indices))
        mismatch_count = 0
        for i in range(sample_checks):
            imgfolder_label = dataset_train.targets[real_train_indices[i]]
            metadata_label = curr_train_targets[i]
            if imgfolder_label != metadata_label:
                mismatch_count += 1
        if mismatch_count > 0:
            print(f"LABEL Mismatch: {mismatch_count}/{sample_checks}")
            print(f"ImageFolder: {dataset_train.class_to_idx}")
        else:
            print(f"Label verification OK")

        
        counts = np.bincount(curr_train_targets)
        ws = 1. / counts
        sample_weights = ws[curr_train_targets]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

        
        unique_c, c_counts = np.unique(curr_train_targets, return_counts=True)
        weights_array = np.ones(8, dtype=np.float32)
        for i, cls in enumerate(unique_c):
            weights_array[cls] = len(curr_train_targets) / (8 * c_counts[i])

        class_weights = torch.tensor(weights_array)
        model = UltimateDermatolog(
            class_weights=class_weights,
            backbone=BACKBONE,
            lr=LR,
            max_epochs=EPOCHS
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath="models/kfold_models/",
            filename=f"{EXPERIMENT_NAME}_fold_{fold+1}",
            monitor="val_loss", mode="min", save_top_k=1, verbose=True
        )
        early_stop = EarlyStopping(monitor="val_loss", patience=7, mode="min")
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            accelerator="cuda",
            devices=1,
            callbacks=[checkpoint_callback, early_stop, lr_monitor],
            precision="16-mixed",
            accumulate_grad_batches=4,
            log_every_n_steps=10
        )

        print(f"FOLD {fold+1} — Starting training with {BACKBONE}")
        trainer.fit(model, train_loader, val_loader)
        print(f"FOLD {fold+1} Completed!")


if __name__ == "__main__":
    main()