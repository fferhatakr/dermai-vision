#Import the necessary libraries
import torch  
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from dataloader.image_dataset import get_data_loaders
from src.training.trainer_core import DermatologLightning 





def main():
    #We define our variables at the outset so that if we wish to make changes later, we only need to make changes in one place.
    EPOCH_NUMBER = 40
    BATCH_SIZE = 32
    DATA_PATH = "Data/images/all_data"



    # The train and validation dataloaders are created by passing DATA_PATH and BATCH_SIZE to the get_data_loaders function.
    print("Data is loading from the factory...")
    train_loader , val_loader = get_data_loaders(DATA_PATH,BATCH_SIZE)



    print(" Penalty points are being calculated...")

    #We are looping through the labels assigned to classes in education.
    train_labels = [label for _, label in train_loader.dataset] 

    #The class weights are calculated using the # compute_class_weight function.
    calculated_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )

    #We call the weight tensor as torch.float and provide the weight calculation as a parameter.
    weight_tensor = torch.FloatTensor(calculated_weights)

    # Launch the Lightning model
    # Class weights are transferred to the loss function within the model.
    print(" Lightning Model Initializing...") 
    lightning_model=DermatologLightning(weight_tensor)

    print(" Setting up Best Checkpoint...")

    # Save the best model based on validation accuracy
    # Prevents overfitting and automatically preserves the most suitable model
    checkpoint_callback=ModelCheckpoint(
        dirpath="models/",
        filename="best_lightning_model",
        monitor="val_acc",
        mode="max",
    )

    # PyTorch Lightning Trainer
    # - Cihaz yerleştirmeyi yönetir (CPU/GPU)
    # - Eğitim döngüsünü yönetir
    # - Geri aramaları otomatik olarak entegre eder
    print(" Trainer is starting the engines!")
    trainer = pl.Trainer(
        max_epochs=EPOCH_NUMBER,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback],
    )

    # Trainer, modeli train ve validation dataloader'ları ile eğitmeye başlar
    trainer.fit(model=lightning_model,train_dataloaders=train_loader,val_dataloaders=val_loader)

# Dosya direkt çalıştırıldığında main() fonksiyonunu başlatır
if __name__ == "__main__":
    main()