import torch
import os
import glob
import sys

sys.path.append(os.getcwd())
from src.training.trainer_core import TripletLightning

def find_latest_checkpoint():
    path1 = glob.glob("lightning_logs/version_*/checkpoints/*.ckpt")
    path2 = glob.glob("models/*.ckpt*") 
    check = path1 + path2
    
    if not check:
        raise FileNotFoundError("No .ckpt files found.")
        
    return max(check, key=os.path.getctime)

def load_model(ckpt_path):
    print(f"Loading model: {ckpt_path}")
    # Initialize model with dummy parameters to load weights safely
    model = TripletLightning.load_from_checkpoint(
        ckpt_path,
        learning_rate=0.001, 
        margin_value=1.0,
        map_location=torch.device('cpu')
    )
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model

def extract_embeddings(model, data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")
        
    full_data = torch.load(data_path)
    embeddings = []
    labels = []
    
    print(f"Processing {len(full_data)} images...")
    
    for i, (img, label) in enumerate(full_data):
        img = img.unsqueeze(0)
        if torch.cuda.is_available():
            img = img.cuda()
            
        with torch.no_grad():
            emb = model(img)
            
        embeddings.append(emb.cpu())
        labels.append(label)

    return embeddings, labels

def save_artifacts(embeddings, labels, save_dir="Data/artifacts"):
    os.makedirs(save_dir, exist_ok=True)
    
    ref_embeddings = torch.cat(embeddings)
    
    # Handle both tensor and integer labels
    if isinstance(labels[0], torch.Tensor):
        ref_labels = torch.stack(labels)
    else:
        ref_labels = torch.tensor(labels)

    torch.save(ref_embeddings, os.path.join(save_dir, "reference_embeddings.pt"))
    torch.save(ref_labels, os.path.join(save_dir, "reference_labels.pt"))
    print(f"Artifacts saved to: {save_dir}")

def main():
    try:
        ckpt_path = find_latest_checkpoint()
        model = load_model(ckpt_path)
        embs, lbls = extract_embeddings(model, "Data/processed/processed_data.pt")
        save_artifacts(embs, lbls)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()