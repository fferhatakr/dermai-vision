from fastapi import FastAPI
import uvicorn
import yaml
from fastapi import UploadFile, File
from PIL import Image
import io
import torch
from torchvision import transforms
import torch.nn.functional as F

from src.training.lightning_model import TripletLightning

def load_ai_model():
    file_path = "configs/inference_config.yaml"
    file_path2 = "configs/train_config.yaml"
    with open(file_path,"r",encoding="utf-8") as file:
        config = yaml.safe_load(file)

    with open(file_path2,"r",encoding="utf-8") as file:
        config2  =yaml.safe_load(file)

    model_path = config["model"]["path"]
    margin_value = config2["model"]["margin_value"]
    
    model = TripletLightning.load_from_checkpoint(
        checkpoint_path=model_path,
        margin_value = margin_value,
        learning_rate = 0.001
    )
    model.eval()

    return model

app = FastAPI(title="DermaScan AI API")
@app.post("/analyze")
async def analyze_image( file:UploadFile = File()):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    transforms_pipeline = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transforms_pipeline(image).unsqueeze(0)

    with torch.no_grad():
        embedding = derma_model(input_tensor)
        distances = torch.cdist(embedding,ref_embeddings)
        _ , indices = torch.topk(distances, k=5, largest=False)

        votes = ref_labels[indices]
        
        
        mode_result = torch.mode(votes)
        majority_vote = mode_result.values.item()
        
        
        confidence = (votes == majority_vote).sum().item() / 5.0
        
       
        is_risky = majority_vote > 0

    return {
        "status": "success",
        "prediction": "Risky" if is_risky else "Normal",
        "confidence": float(confidence), 
        "message": "The analysis is complete. Please consult a specialist."
    }

derma_model = load_ai_model()
ref_embeddings = torch.load("reference_embeddings.pt", map_location="cpu")
ref_labels = torch.load("reference_labels.pt", map_location="cpu")

@app.get("/")
def  application():
    return {"mesaj":"Hello User"}

if __name__ == "__main__":
    uvicorn.run("src.api.main:app",host="127.0.0.1",port=8000,reload=True)