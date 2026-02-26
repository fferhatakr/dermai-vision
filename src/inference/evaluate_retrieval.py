import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import yaml
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.training.lightning_model import TripletLightning

def main():
    print("🚀 Evaluation is starting...")

    
    with open("configs/train_config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    
    model_path = os.path.join(config['model']['checkpoint_dir'], config['model']['checkpoint_name'])
    print(f"🧠 Loading Model: {model_path}")
    
    
    model = TripletLightning.load_from_checkpoint(
        model_path,
        learning_rate=config['training']['learning_rate'],
        margin_value=config['model']['margin_value']
    )
    model.eval() 
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    

    dataset = datasets.ImageFolder(root=config['data']['data_path'], transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    print("🔍 Gallery (Database) is being extracted... This process may take a while.")
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(model.device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.extend(labels.numpy())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.tensor(all_labels)
    
    print(f"✅ Gallery Ready! Total Images: {all_embeddings.size(0)}, Vector Size: {all_embeddings.size(1)}")


    print("🎯 Top-5 Success Rate Calculating...")
    

    all_embeddings_norm = F.normalize(all_embeddings, p=2, dim=1)
    similarity_matrix = torch.mm(all_embeddings_norm, all_embeddings_norm.t())

    top_k = 5
    correct_hits = 0

    for i in range(len(all_labels)):

        similarity_matrix[i, i] = -1.0 
        

        top_k_indices = torch.topk(similarity_matrix[i], top_k).indices
        top_k_labels = all_labels[top_k_indices]
        
 
        if all_labels[i] in top_k_labels:
            correct_hits += 1

    top5_accuracy = (correct_hits / len(all_labels)) * 100
    print(f"\n🏆 RESULT: When a new patient arrives, the model has a %{top5_accuracy:.2f} probability of correctly identifying the disease within the first 5 results!")

    torch.save(all_embeddings, "reference_embeddings.pt")
    torch.save(all_labels, "reference_labels.pt")
    print("💾 The reference database API has been successfully saved to disk!")
if __name__ == "__main__":
    main()