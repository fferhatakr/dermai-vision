import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.training.trainer_core import TripletLightning
from torch.utils.data import DataLoader , Subset
from src.dataloader.image_dataset import TripletDermaDataset
import yaml 
import torch
import random


torch.set_float32_matmul_precision('medium')
pl.seed_everything(42, workers=True)
def main():
    with open("configs/train_config.yaml","r",encoding = "utf-8") as file:
        config = yaml.safe_load(file)

    EPOCH_NUMBER = config['training']['epoch_number']
    BATCH_SIZE = config['data']['batch_size']
    DATA_PATH = config['data']['data_path']


    train_base = TripletDermaDataset(DATA_PATH, is_train=True)
    val_base = TripletDermaDataset(DATA_PATH, is_train=False)

    
    total_len = len(train_base)
    indices = list(range(total_len))
    
    
    random.seed(42)
    random.shuffle(indices)

    split = int(0.8 * total_len)
    train_idx = indices[:split]
    val_idx = indices[split:]

    
    train_subset = Subset(train_base, train_idx)
    val_subset = Subset(val_base, val_idx)
    

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
        )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    
    triplet_model = TripletLightning(
        margin_value=config['model']['margin_value'],
        learning_rate = config['training']['learning_rate']
    )
    
    


    checkpoint_callback=ModelCheckpoint(
        dirpath=config['model']['checkpoint_dir'],
        filename = config['model']['checkpoint_name'],
        monitor="val_loss",
        mode="min",
    )

    
    trainer = pl.Trainer(
        max_epochs=EPOCH_NUMBER,
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback],
    )


    trainer.fit(
        model=triplet_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
        )

if __name__ == "__main__":
    main()