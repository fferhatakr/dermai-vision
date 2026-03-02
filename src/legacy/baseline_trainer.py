#Import the necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torchvision import models
import sys
import os 
from dataloader.image_dataset import get_data_loaders
from architectures.vision_model import SkinCancerMobileNet
from utils.helpers import matrix_draw

#It shows the file path on our computer. You may not need to do it.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))) 



def main():
    
    EPOCH_NUMBER = 100
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 32
    DATA_PATH = "Data/images/all_data"  

    #We call our get_Data_loaders function for train_loader and val_loader and provide the necessary parameters.   
    print(" Data is loading from the factory...")
    train_loader, val_loader = get_data_loaders(DATA_PATH, BATCH_SIZE)

    #If there is a graphics card available in our system, it will use that; otherwise, it will use the processor.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" VIP Room Used: {device}")

    
    #You can think of it as if we have fitted our model to the existing engine.
    model = SkinCancerMobileNet().to(device)

    
    print(" Penalty points are being calculated...")
    train_labels = [label for _, label in train_loader.dataset] 
    calculated_weights = compute_class_weight(
        class_weight="balanced",  #We are specifying the penalty section.
        classes=np.unique(train_labels),
        y=train_labels 
    )
    
    weight_tensor = torch.FloatTensor(calculated_weights).to(device)
    
    #Making definitions for calculating the loss function
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    #
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

   
    print(" Engines are running! Training has begun...")
    best_val_accuracy = 0.0

    for epoch in range(EPOCH_NUMBER):
        model.train() 
        train_loss = 0
        train_accuracy = 0
        
        for image, label in train_loader:
            image, label = image.to(device), label.to(device)
            
            predict = model(image)
            loss = criterion(predict, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = torch.argmax(predict, dim=1)
            batch_correct = (predictions == label).sum().item()
            train_accuracy += batch_correct

        mean_train_loss = train_loss / len(train_loader)
        percentage_of_train_success = (train_accuracy / len(train_loader.dataset)) * 100

    
        print(" Training is complete, moving on to the testing phase...")
        
        model.eval() 
        val_loss = 0
        val_accuracy = 0
        actual_tags = []
        model_predictions  = []
    
        with torch.no_grad():
            for val_image, val_label in val_loader:
                val_image, val_label = val_image.to(device), val_label.to(device)
                val_predict = model(val_image)
                v_loss = criterion(val_predict, val_label)

                val_loss += v_loss.item()
                v_predictions = torch.argmax(val_predict, dim=1)
                v_batch_correct = (v_predictions == val_label).sum().item()
                val_accuracy += v_batch_correct

                model_predictions .extend(v_predictions.cpu().numpy())
                actual_tags.extend(val_label.cpu().numpy())

            mean_val_loss = val_loss / len(val_loader)
            percentage_of_val_success = (val_accuracy / len(val_loader.dataset)) * 100

            print(f"Epoch {epoch+1}/{EPOCH_NUMBER} | Train Loss: {mean_train_loss:.4f}, Train Acc: %{percentage_of_train_success:.2f} | Val Loss: {mean_val_loss:.4f}, Val Acc: %{percentage_of_val_success:.2f}")
            if percentage_of_val_success > best_val_accuracy:
                print(f" NEW HIGH SCORE! (%{percentage_of_val_success:.2f}). ")
                best_val_accuracy = percentage_of_val_success
                torch.save(model.state_dict(), "models/dermatolog_v4.2.pth")
                best_actual_tags = actual_tags
                best_model_predictions = model_predictions

    print(" Training completed! The complexity matrix of the best model is being plotted...")
    matrix_draw(best_actual_tags, best_model_predictions, title="DermaScan V4.2 - Production Test")


if __name__ == "__main__":
        main()