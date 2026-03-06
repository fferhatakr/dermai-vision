from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
from PIL import Image
import io
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np
import cv2
import base64
import os
import sys
from src.engine.train_class_v2 import UltimateDermatolog

sys.path.append(os.getcwd())
from src.training.trainer_core import DermatologLightning

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

def load_ai_models():
    model_path = "models/production/classifier_v3_best.ckpt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ERROR: {model_path} not found!")
    
    vision_model = DermatologLightning.load_from_checkpoint(model_path, class_weights=None, strict=False)
    vision_model.to(DEVICE)
    vision_model.eval()

    nlp_path = "models/production/nlp"
    tokenizer = DistilBertTokenizer.from_pretrained(nlp_path)
    text_model = DistilBertForSequenceClassification.from_pretrained(nlp_path)
    text_model.to(DEVICE)
    text_model.eval()

    return vision_model, tokenizer, text_model

vision_model, tokenizer, text_model = load_ai_models()

app = FastAPI(title="DermaScan AI API - V2 Ultimate")

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    text: str = Form(default="No symptoms provided"),
    needs_heatmap: bool = Form(False)
):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_orig = base_transform(image).unsqueeze(0).to(DEVICE)
    img_h = base_transform(transforms.functional.hflip(image)).unsqueeze(0).to(DEVICE)
    img_v = base_transform(transforms.functional.vflip(image)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        p1 = F.softmax(vision_model(img_orig), dim=1)
        p2 = F.softmax(vision_model(img_h), dim=1)
        p3 = F.softmax(vision_model(img_v), dim=1)
        avg_probs = (p1 + p2 + p3) / 3.0

    img_risk_score = (avg_probs[0][0] + avg_probs[0][1] + avg_probs[0][4]).item()

    encoded_heatmap = ""
    if needs_heatmap:
        img_orig.requires_grad = True
        heat_map = vision_model.generate_gradcam(img_orig)
        
        heatmap_np = heat_map.squeeze().cpu().detach().numpy()
        heatmap_np = cv2.resize(heatmap_np, (image.size[0], image.size[1]))
        heatmap_np = (heatmap_np * 255).astype(np.uint8)
        color_heatmap = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)
        
        _, heatmap_png = cv2.imencode(".png", color_heatmap)
        encoded_heatmap = base64.b64encode(heatmap_png).decode('utf-8')

    nlp_inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        nlp_outputs = text_model(**nlp_inputs)
        nlp_probs = F.softmax(nlp_outputs.logits, dim=1)
        nlp_risk_score = nlp_probs[0][1].item()

    hybrid_score = (img_risk_score * 0.7) + (nlp_risk_score * 0.3)

    return {
        "status": "success",
        "prediction": "Risky" if hybrid_score >= 0.5 else "Normal",
        "confidence": float(hybrid_score),
        "scores": {
            "image": float(img_risk_score),
            "text": float(nlp_risk_score)
        },
        "message": f"Analysis Complete. (Image Risk: {img_risk_score:.2f}, Text Risk: {nlp_risk_score:.2f})",
        "heatmap_base64": encoded_heatmap
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)