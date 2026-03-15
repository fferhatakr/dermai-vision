import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
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
            logits, 
            labels, 
            weight=self.class_weights, 
            label_smoothing=0.1, 
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** 2.0) * ce_loss
        return focal_loss.mean()

    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        
       
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5,    
            patience=3,    
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
    

def main():

    KFOLD_DATA_PATH = "Data/processed/kfold_data"
    BATC_SIZE = 16
    EPOCHS = 25
    K_FOLDS = 5

    dataset_train = datasets.ImageFolder(KFOLD_DATA_PATH,transform=train_transforms)
    dataset_val = datasets.ImageFolder(KFOLD_DATA_PATH,transform=val_transforms)

    targets = np.array(dataset_train.targets)
    classes = np.unique(targets)

    kfold = StratifiedKFold(n_splits=K_FOLDS,shuffle=True,random_state=42)

    for fold,(train_idx,val_idx) in enumerate(kfold.split(np.zeros(len(targets)),targets)):
        if fold < 2:
            print(f"Fold {fold+1} already trained, skipped")
            continue
        

        train_sub = Subset(dataset_train,train_idx)
        val_sub = Subset(dataset_val,val_idx)

        current_train_targets = targets[train_idx]
        class_counts = np.bincount(current_train_targets)
        class_weights_sampler = 1. / class_counts
        sample_weights = class_weights_sampler[current_train_targets]


        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = DataLoader(
            train_sub,
            batch_size=BATC_SIZE,
            sampler=sampler,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_sub,
            batch_size=BATC_SIZE,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )


        weights = compute_class_weight(class_weight='balanced',classes=classes,y=current_train_targets)
        class_weights = torch.tensor(weights, dtype=torch.float32)

        model = UltimateDermatolog(class_weights=class_weights)

        checkpoint_callback = ModelCheckpoint(
            dirpath="models/kfold_models/", 
            filename=f"ultimate_v5_fold_{fold+1}",
            monitor="val_loss", mode="min", save_top_k=1, verbose=True
        )

        early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        # swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)

        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            accelerator="cuda",
            devices=1,
            callbacks=[checkpoint_callback, early_stop, lr_monitor],
            precision="16-mixed",
            accumulate_grad_batches=4, 
            log_every_n_steps=10
        )

        trainer.fit(model, train_loader, val_loader)
        print(f" FOLD {fold+1} Success\n")

if __name__ == "__main__":
    main()