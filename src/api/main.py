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

sys.path.append(os.getcwd())
from src.training.trainer_core import DermatologLightning

CKPT_PATH = "models/ultimate_isic_PRO_v4.ckpt"
NLP_PATH = "models/production/nlp"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['0_mel', '1_nv', '2_bcc', '3_ak', '4_bkl', '5_df', '6_vasc', '7_scc']

app = FastAPI(title="DermaScan AI API")

vision_model = None
tokenizer = None
text_model = None

@app.on_event("startup")
def load_ai_models():
    global vision_model, tokenizer, text_model
    if not os.path.exists(CKPT_PATH): raise FileNotFoundError(f"No model: {CKPT_PATH}")
    global vision_model, lightning_model

    lightning_model = DermatologLightning.load_from_checkpoint(CKPT_PATH, strict=False)
    vision_model = lightning_model.model

    lightning_model.to(DEVICE).eval()
    vision_model.to(DEVICE).eval()
    
    try:
        if os.path.exists(NLP_PATH):
            tokenizer = DistilBertTokenizer.from_pretrained(NLP_PATH)
            text_model = DistilBertForSequenceClassification.from_pretrained(NLP_PATH)
            text_model.to(DEVICE).eval()
    except: pass

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), text: str = Form(default=""), needs_heatmap: bool = Form(False)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = vision_model(input_tensor)
        probs = F.softmax(logits, dim=1)
        
    probs_np = probs.cpu().numpy()[0]
    
    prob_mel = float(probs_np[0])
    prob_bcc = float(probs_np[2]) 
    prob_scc = float(probs_np[7])
    
    top_idx = int(np.argmax(probs_np))
    diagnosis = CLASSES[top_idx]
    confidence = float(probs_np[top_idx])
    
    is_risky = False
    if prob_mel > 0.18:
        diagnosis = "MEL (Melanoma)"
        confidence = prob_mel
        is_risky = True
    elif prob_bcc > 0.25:
        diagnosis = "BCC (Basal Cell Carcinoma)"
        confidence = prob_bcc
        is_risky = True
    elif prob_scc > 0.25:
        diagnosis = "SCC (Squamous Cell Carcinoma)"
        confidence = prob_scc
        is_risky = True
    elif top_idx in [0, 2, 3, 7]:
        is_risky = True

    nlp_score = 0.0
    if text_model is not None and tokenizer is not None and text:
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)
            with torch.no_grad():
                outputs = text_model(**inputs)
                nlp_probs = F.softmax(outputs.logits, dim=1)
                nlp_score = float(nlp_probs[0][1].item())
        except Exception as e:
            print(f"NLP Error: {e}")
            nlp_score = 0.0

    
    hybrid_score = (confidence * 0.8) + (nlp_score * 0.2)

    encoded_heatmap = ""
    if needs_heatmap:
        try:
            input_tensor.requires_grad = True
            
            hm = lightning_model.generate_gradcam(input_tensor)
            

            if isinstance(hm, torch.Tensor):
                hm = hm.squeeze().detach().cpu().numpy()
            

            hm = cv2.resize(hm, (image.size[0], image.size[1]))

            h, w = hm.shape
            margin_h = int(h * 0.10) 
            margin_w = int(w * 0.10)
            hm[:margin_h, :] = 0  
            hm[-margin_h:, :] = 0 
            hm[:, :margin_w] = 0  
            hm[:, -margin_w:] = 0 


            hm_uint8 = np.uint8(255 * hm)
            hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
            
            _, buf = cv2.imencode(".png", hm_color)
            encoded_heatmap = base64.b64encode(buf).decode("utf-8")

        except Exception as e:
            print(f"Heatmap Encode Error: {e}")
            
            pass
    all_probs = {}
    for i, class_name in enumerate(CLASSES):

        clean_name = class_name.split('_')[1].upper()
        if clean_name == "MEL": 
            clean_name = "MEL (Melanoma)"
        elif clean_name == "NV": 
            clean_name = "NV (Nevus)"
        elif clean_name == "BCC": 
            clean_name = "BCC (Basal Cell Carcinoma)"
        
        all_probs[clean_name] = float(probs_np[i])
    

    sorted_probs = dict(sorted(all_probs.items(), key=lambda item: item[1], reverse=True))

    return {
        "prediction": "Risky" if is_risky else "Benign",
        "diagnosis": diagnosis,
        "confidence": confidence,
        "hybrid_score": hybrid_score,
        "all_probabilities": sorted_probs,
        "scores": {
            "image_raw": confidence, 
            "text": nlp_score,
            "hybrid": hybrid_score,
            "severity": min(100, (hybrid_score / 0.25) * 50)
        },
        "message": f"RAW: Mel %{prob_mel*100:.1f} | BCC %{prob_bcc*100:.1f}",
        "heatmap_base64": encoded_heatmap
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)