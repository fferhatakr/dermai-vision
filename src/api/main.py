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
import onnxruntime as ort
sys.path.append(os.getcwd()) 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_ai_model():
    file_path = glob.glob("lightning_logs/version_*/checkpoints/*.ckpt")
    file_path2 = glob.glob("models/*.ckpt*")

    checkpoints = file_path + file_path2
    
    if not checkpoints:
        raise FileNotFoundError("ERROR: No .ckpt model file found!")
    
    latest_ckpt = max(checkpoints, key=os.path.getctime)
    print(f" Uploaded Model: {latest_ckpt}")

    detailed_model = TripletLightning.load_from_checkpoint(
        checkpoint_path=latest_ckpt,
        margin_value=1.0,      
        learning_rate=0.001,   
        map_location=torch.device('cpu') 
    )
    detailed_model.to(DEVICE)
    detailed_model.eval()

    nlp_model_path = "models/nlp_v1"
    if not os.path.exists(nlp_model_path):
         raise FileNotFoundError(f" Error: '{nlp_model_path}'")

    tokenizer = DistilBertTokenizer.from_pretrained(nlp_model_path)
    text_model = DistilBertForSequenceClassification.from_pretrained(nlp_model_path)
    text_model.to(DEVICE)
    text_model.eval()


    #Fast Model:Optional
    onnx_path = "models/derma_vision_large_v1.onnx"
    if not os.path.exists(onnx_path):
        print(f"Warning: ONNX model not found.")
        fast_model = None
        
    else:
        print(f"Fast Model:{onnx_path}")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        fast_model = ort.InferenceSession(onnx_path, providers=providers)

    return detailed_model,tokenizer ,text_model ,fast_model


derma_model,tokenizer,text_model,fast_model = load_ai_model()
path = "Data/artifacts/reference_embeddings.pt"
path2= "Data/artifacts/reference_labels.pt"
ref_embeddings = torch.load(path, map_location=DEVICE)
ref_labels = torch.load(path2, map_location=DEVICE)

app = FastAPI(title="DermaScan AI API")

@app.post("/analyze")
async def analyze_image( 
    file:UploadFile = File(),
    text: str = Form(default="No symptoms provided"),
    needs_heatmap: bool = Form(False)
    ):

    #Image Processing -->>Common area required for the 2nd Rule to function:
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    transforms_pipeline = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transforms_pipeline(image).unsqueeze(0).to(DEVICE)

    #We are initialising the variables (so they are not left empty)
    query_embedding = None
    encoded_heatmap = ""

    if needs_heatmap or fast_model is None:
        
        with torch.no_grad():
            query_embedding = derma_model(input_tensor)

        if needs_heatmap:
            input_tensor.requires_grad = True
            heat_map = derma_model.generate_gradcam(input_tensor) 
            
            
            tensor_heatmap = heat_map.squeeze().cpu().detach().numpy()
            tensor_heatmap = (tensor_heatmap - tensor_heatmap.min()) / (tensor_heatmap.max() - tensor_heatmap.min() + 1e-8)
            
            extended_heatmap = cv2.resize(tensor_heatmap, (224,224)) 
            blawhite_heatmap = (extended_heatmap * 255).astype(np.uint8)
            color_heatmap = cv2.applyColorMap(blawhite_heatmap, cv2.COLORMAP_JET)
            
            _, heatmap_png = cv2.imencode(".png", color_heatmap)
            encoded_heatmap = base64.b64encode(heatmap_png).decode('utf-8')
    else:
        
        onnx_input = input_tensor.cpu().numpy()
        ort_outs = fast_model.run(None, {"input_image": onnx_input})
        query_embedding = torch.from_numpy(ort_outs[0]).to(DEVICE)
            

    with torch.no_grad():
            
            distances = torch.cdist(query_embedding, ref_embeddings)
            _, indices = torch.topk(distances, k=5, largest=False)
            votes = ref_labels[indices.cpu()]
            
            mode_result = torch.mode(votes)
            majority_vote = mode_result.values.item()
            risky_labels = {0, 1, 4} 
            is_risky = majority_vote in risky_labels
            confidence = (votes == majority_vote).sum().item() / 5.0

            
            nlp_inputs = tokenizer(
                text, return_tensors="pt", truncation=True, padding=True, max_length=128
            ).to(DEVICE)
            nlp_outputs = text_model(**nlp_inputs)
            nlp_probs = F.softmax(nlp_outputs.logits, dim=1)
            nlp_risk_score = nlp_probs[0][1].item()

            
            img_risk_score = float(confidence) if is_risky else (1.0 - float(confidence))
            hybrid_score = (img_risk_score * 0.7) + (nlp_risk_score * 0.3)
            final_is_risky = hybrid_score >= 0.5

           
    return {
        "status": "success",
        "prediction": "Risky" if hybrid_score >= 0.5 else "Normal",
        "confidence": float(hybrid_score),
        "scores": {
            "image": float(img_risk_score), 
            "text": float(nlp_risk_score)
        },
        "message": f"Analysis Complete. (Image: {img_risk_score:.2f}, Text: {nlp_risk_score:.2f})",
        "heatmap_base64": encoded_heatmap, 
        "model_used": "PyTorch" if (needs_heatmap or fast_model is None) else "ONNX"
    }
@app.get("/")
def  application():
    return {"message":"Hello User"}

if __name__ == "__main__":
    uvicorn.run("src.api.main:app",host="127.0.0.1",port=8000,reload=True)