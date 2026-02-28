import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.training.trainer_core import TripletLightning
from torch.utils.data import DataLoader
from src.dataloader.image_dataset import TripletDermaDataset
import yaml

pl.seed_everything(42, workers=True)




def main():
    with open("configs/train_config.yaml","r",encoding = "utf-8") as file:
        config = yaml.safe_load(file)

    EPOCH_NUMBER = config['training']['epoch_number']
    BATCH_SIZE = config['data']['batch_size']
    DATA_PATH = config['data']['data_path']



    
    
    triplet_dataset = TripletDermaDataset(DATA_PATH)
    train_loader = DataLoader(
        triplet_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
        )

    
    triplet_model = TripletLightning(
        margin_value=config['model']['margin_value'],
        learning_rate = config['training']['learning_rate']
    )
    
    


    checkpoint_callback=ModelCheckpoint(
        dirpath=config['model']['checkpoint_dir'],
        filename = config['model']['checkpoint_name'],
        monitor="train_loss",
        mode="min",
    )

    
    trainer = pl.Trainer(
        max_epochs=EPOCH_NUMBER,
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback],
    )


    trainer.fit(model=triplet_model,train_dataloaders=train_loader)

if __name__ == "__main__":
    main()