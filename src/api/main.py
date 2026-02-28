from fastapi import FastAPI
import uvicorn
import yaml
from fastapi import UploadFile, File,Form
from PIL import Image
import io
import torch
from torchvision import transforms
import torch.nn.functional as F
from transformers import DistilBertTokenizer , DistilBertForSequenceClassification
import numpy as np
import cv2
import base64
import os   
import glob 
import sys  
from src.training.trainer_core import TripletLightning
sys.path.append(os.getcwd()) 




def load_ai_model():
    file_path = glob.glob("lightning_logs/version_*/checkpoints/*.ckpt")
    file_path2 = glob.glob("models/*.ckpt*")

    checkpoints = file_path + file_path2
    
    if not checkpoints:
        
        raise FileNotFoundError("ERROR: No .ckpt model file found!")
    
    latest_ckpt = max(checkpoints, key=os.path.getctime)
    print(f" Uploaded Model: {latest_ckpt}")

    model = TripletLightning.load_from_checkpoint(
        checkpoint_path=latest_ckpt,
        margin_value=1.0,      
        learning_rate=0.001,   
        map_location=torch.device('cpu') 
    )
    model.eval()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    text_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

    return model,tokenizer ,text_model

app = FastAPI(title="DermaScan AI API")
@app.post("/analyze")
async def analyze_image( file:UploadFile = File(),text: str = Form(default="No symptoms provided")):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    transforms_pipeline = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transforms_pipeline(image).unsqueeze(0)

    with torch.no_grad():
        derma_model.eval()
        embedding = derma_model(input_tensor)
        distances = torch.cdist(embedding,ref_embeddings)
        _ , indices = torch.topk(distances, k=5, largest=False)

        votes = ref_labels[indices]
        
        
        mode_result = torch.mode(votes)
        majority_vote = mode_result.values.item()
        is_risky = majority_vote > 0
        
        confidence = (votes == majority_vote).sum().item() / 5.0

        nlp_inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
        )
        nlp_outputs = text_model(**nlp_inputs)
        nlp_probs = F.softmax(nlp_outputs.logits,dim=1)
        nlp_risk_score = nlp_probs[0][1].item()

        img_risk_score = float(confidence) if is_risky else (1.0 - float(confidence))
        hybrid_score = (img_risk_score * 0.6) + (nlp_risk_score * 0.4)
        final_is_risky = hybrid_score > 0.5

    heat_map = derma_model.generate_gradcam(input_tensor)
    tensor_heatmap = heat_map.squeeze().cpu().detach().numpy()
    extended_heatmap = cv2.resize(tensor_heatmap,(224,224))
    blawhite_heatmap = (extended_heatmap*255)
    unmarked_heatmap = np.uint8(blawhite_heatmap)

    color_heatmap = cv2.applyColorMap(unmarked_heatmap,cv2.COLORMAP_JET)
    _,heatmap_png = cv2.imencode(".png",color_heatmap)
    y = heatmap_png
    encode_heatmap = base64.b64encode(y)
    encoded_heatmap = encode_heatmap.decode('utf-8')



       
        

    return {
        "status": "success",
        "prediction": "Risky" if final_is_risky else "Normal",
        "confidence": float(hybrid_score), 
        "message": f"Hybrid Analysis Complete. (Image Score: {img_risk_score:.2f}, Text Score: {nlp_risk_score:.2f})",
        "heatmap_base64": encoded_heatmap
    }

derma_model,tokenizer,text_model, = load_ai_model()
path = "Data/artifacts/reference_embeddings.pt"
path2= "Data/artifacts/reference_labels.pt"
ref_embeddings = torch.load(path, map_location="cpu")
ref_labels = torch.load(path2, map_location="cpu")

@app.get("/")
def  application():
    return {"message":"Hello User"}

if __name__ == "__main__":
    uvicorn.run("src.api.main:app",host="127.0.0.1",port=8000,reload=True)