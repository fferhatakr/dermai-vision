import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor,StochasticWeightAveraging
from sklearn.utils.class_weight import compute_class_weight
import sys
import os
import torch.nn.functional as F
sys.path.append(os.getcwd())
from src.dataloader.image_dataset import get_data_loaders
from src.training.trainer_core import DermatologLightning


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
    print("TRAINING STARTING: Weighted Loss + LR Scheduler")
    
    TRAIN_PATH = "Data/processed/train"
    VAL_PATH = "Data/processed/val"
    BATCH_SIZE = 16
    EPOCHS = 25 
    
    train_loader, val_loader = get_data_loaders(TRAIN_PATH,VAL_PATH, BATCH_SIZE,)
    
    
    train_targets = train_loader.dataset.targets 
    classes = np.unique(train_targets)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_targets)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    
    model = UltimateDermatolog(class_weights=class_weights)

    
    checkpoint_callback = ModelCheckpoint(
        dirpath="models/",
        filename="ultimate_isic_PRO_v4",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True
    )
    
    
    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="cuda", 
        devices=1,
        callbacks=[checkpoint_callback, early_stop, lr_monitor,swa_callback],
        precision="16-mixed",
        accumulate_grad_batches=4,
        log_every_n_steps=10
    )

    
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()

