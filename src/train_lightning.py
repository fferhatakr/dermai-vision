import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from datalar.dataset import get_data_loaders
from lightning_model import DermatologLightning 





def main():
    #Değişkenleri tek tek tanımlıyoruz
    EPOCH_NUMBER = 40
    BATCH_SIZE = 32
    DATA_PATH = "Data/train"

    print("Data is loading from the factory...")
    # get_data_loaders fonksiyonuna DATA_PATH ve BATCH_SIZE verilerek
    # train ve validation dataloader'ları oluşturuluyor
    train_loader , val_loader = get_data_loaders(DATA_PATH,BATCH_SIZE)

    print("⚖️ Penalty points are being calculated...")
    #Eğitim etiketleri
    train_labels = [label for _, label in train_loader.dataset] 
    # compute_class_weight fonksiyonu ile sınıf ağırlıkları hesaplanıyor
    calculated_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )


    weight_tensor = torch.FloatTensor(calculated_weights)

    # Özel Lightning modelini başlat
    # Sınıf ağırlıkları model içindeki kayıp fonksiyonuna aktarılır
    print("⚡ Lightning Model Initializing...") 
    lightning_model=DermatologLightning(weight_tensor)

    print("💾 Setting up Best Checkpoint...")

    # Doğrulama doğruluğuna göre en iyi modeli kaydet
    # Aşırı uyumu önler ve otomatik olarak en uygun modeli korur
    checkpoint_callback=ModelCheckpoint(
        dirpath="models/",
        filename="best_lightning_model",
        monitor="val_acc",
        mode="max",
    )

    print("🔥 Trainer is starting the engines!")
    # PyTorch Lightning Trainer
    # - Cihaz yerleştirmeyi yönetir (CPU/GPU)
    # - Eğitim döngüsünü yönetir
    # - Geri aramaları otomatik olarak entegre eder
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