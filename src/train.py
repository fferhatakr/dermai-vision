import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


from dataset import get_data_loaders
from model import SkinCancerModelV2
from utils import matrix_draw

def main():
    
    EPOCH_NUMBER = 15
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 32
    DATA_PATH = "Data/train"  

    
    print("📦 Data is loading from the factory...")
    train_loader, val_loader = get_data_loaders(DATA_PATH, BATCH_SIZE)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ VIP Room Used: {device}")

    
    model = SkinCancerModelV2().to(device)

    
    print("⚖️ Penalty points are being calculated...")
    train_labels = [label for _, label in train_loader.dataset] 
    calculated_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )
    weight_tensor = torch.FloatTensor(calculated_weights).to(device)
    
    
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

   
    print("🔥 Engines are running! Training has begun...")
    
    model.train() 
    for epoch in range(EPOCH_NUMBER):
        total_loss = 0
        total_accuary = 0
        
        for image, label in train_loader:
            image, label = image.to(device), label.to(device)
            
            predict = model(image)
            loss = criterion(predict, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = torch.argmax(predict, dim=1)
            batch_correct = (predictions == label).sum().item()
            total_accuary += batch_correct

        mean_loss = total_loss / len(train_loader)
        percentage_of_success = (total_accuary / len(train_loader.dataset)) * 100

        print(f"Epoch {epoch+1}/{EPOCH_NUMBER} completed. Mean Loss: {mean_loss:.4f}, Percentage of Success: %{percentage_of_success:.2f}")

    
    print("🧪 Training is complete, moving on to the testing phase...")
    
    model.eval() 
    actual_tags = []
    model_predictions  = []
    
    with torch.no_grad():
        total_loss = 0
        total_accuary = 0

        for image, label in val_loader:
            image, label = image.to(device), label.to(device)
            predict = model(image)
            loss = criterion(predict, label)

            total_loss += loss.item()
            predictions = torch.argmax(predict, dim=1)
            batch_correct = (predictions == label).sum().item()
            total_accuary += batch_correct

            model_predictions .extend(predictions.cpu().numpy())
            actual_tags.extend(label.cpu().numpy())

        mean_loss = total_loss / len(val_loader)
        percentage_of_success = (total_accuary / len(val_loader.dataset)) * 100

        print(f"🎉 TEST RESULT | Mean Loss: {mean_loss:.4f} | Success Rate: %{percentage_of_success:.2f}")

    
    torch.save(model.state_dict(), "models/dermatolog_v3_1.pth")
    print("✅ Model weights 'models/dermatolog_v3_1.pth' olarak kaydedildi!")
    matrix_draw(actual_tags, model_predictions , title="Cepteki Dermatolog V3.1 - Production Test")


if __name__ == "__main__":
    main()