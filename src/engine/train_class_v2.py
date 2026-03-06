import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from sklearn.utils.class_weight import compute_class_weight
import sys
import os


sys.path.append(os.getcwd())
from src.dataloader.image_dataset import get_data_loaders
from src.training.trainer_core import DermatologLightning


class UltimateDermatolog(DermatologLightning):
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=2, 
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
    print("ULTIMATE TRAINING STARTING: Weighted Loss + LR Scheduler")
    
    
    DATA_PATH = "Data/images/all_data"
    BATCH_SIZE = 32
    EPOCHS = 15 
    
    # Load data
    print("Loading data...")
    train_loader, val_loader = get_data_loaders(DATA_PATH, BATCH_SIZE)
    
    
    print("Calculating weights...")
    train_targets = []
    for _, label in train_loader.dataset:
        train_targets.append(label)
    
    classes = np.unique(train_targets)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_targets)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    
    print(f"Class Weights: {class_weights}")

   
    model = UltimateDermatolog(class_weights=class_weights)

    
    checkpoint_callback = ModelCheckpoint(
        dirpath="models/",
        filename="ultimate_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )
    
    
    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="cuda", 
        devices=1,
        callbacks=[checkpoint_callback, early_stop, lr_monitor],
        num_sanity_val_steps=0 
    )

    
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()