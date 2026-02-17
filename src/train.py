import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Kendi yazdığımız dosyaları (modülleri) içeri alıyoruz!
from dataset import veri_yukleyicileri_getir
from model import SkinCancerModelV2
from utils import matris_cizdir

def main():
    
    EPOCH_SAYISI = 15
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 32
    VERI_YOLU = "Data/train"  

    
    print("📦 Veriler fabrikadan yükleniyor...")
    train_loader, val_loader = veri_yukleyicileri_getir(VERI_YOLU, BATCH_SIZE)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Kullanılan VIP Oda: {device}")

    
    model = SkinCancerModelV2().to(device)

    
    print("⚖️ Ceza puanları hesaplanıyor...")
    train_labels = [label for _, label in train_loader.dataset] 
    hesaplanan_agirliklar = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )
    weight_tensor = torch.FloatTensor(hesaplanan_agirliklar).to(device)
    
    
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

   
    print("🔥 Motorlar Ateşleniyor! Eğitim Başladı...")
    
    model.train() # Modeli eğitim moduna al
    for epoch in range(EPOCH_SAYISI):
        total_loss = 0
        total_accuary = 0
        
        for image, label in train_loader:
            image, label = image.to(device), label.to(device)
            
            predict = model(image)
            hata = criterion(predict, label)

            optimizer.zero_grad()
            hata.backward()
            optimizer.step()

            total_loss += hata.item()
            tahmin_edilenler = torch.argmax(predict, dim=1)
            paket_dogrusu = (tahmin_edilenler == label).sum().item()
            total_accuary += paket_dogrusu

        ortalama_hata = total_loss / len(train_loader)
        yuzdelik_basari = (total_accuary / len(train_loader.dataset)) * 100

        print(f"Epoch {epoch+1}/{EPOCH_SAYISI} tamamlandı. Ortalama Hata: {ortalama_hata:.4f}, Yüzdelik Başari: %{yuzdelik_basari:.2f}")

    
    print("🧪 Eğitim bitti, test aşamasına geçiliyor...")
    
    model.eval() 
    gercek_etiketler = []
    modelin_tahminleri = []
    
    with torch.no_grad():
        total_loss = 0
        total_accuary = 0

        for image, label in val_loader:
            image, label = image.to(device), label.to(device)
            predict = model(image)
            hata = criterion(predict, label)

            total_loss += hata.item()
            tahmin_edilenler = torch.argmax(predict, dim=1)
            paket_dogrusu = (tahmin_edilenler == label).sum().item()
            total_accuary += paket_dogrusu

            modelin_tahminleri.extend(tahmin_edilenler.cpu().numpy())
            gercek_etiketler.extend(label.cpu().numpy())

        ortalama_hata = total_loss / len(val_loader)
        yuzdelik_basari = (total_accuary / len(val_loader.dataset)) * 100

        print(f"🎉 TEST SONUCU | Ortalama Hata: {ortalama_hata:.4f} | Başarı Oranı: %{yuzdelik_basari:.2f}")

    
    matris_cizdir(gercek_etiketler, modelin_tahminleri, baslik="Cepteki Dermatolog V3.1 - Prodüksiyon Testi")


if __name__ == "__main__":
    main()