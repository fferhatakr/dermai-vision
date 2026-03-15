from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
from PIL import Image
import io
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import pandas as pd
import src.api.models as ai_models
from src.api.models import DEVICE, CLASSES
from src.api.inference import apply_tta, apply_vignette
from src.api.gradcam import generate_heatmap
from src.api.schemas import AnalysisResponse



app = FastAPI(title="DermaScan AI - Full Debug Meta-Engine")


@app.on_event("startup")
def startup():
    ai_models.load_ai_models()



@app.post("/analyze",response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...), 
    age: float = Form(...), 
    sex: str = Form(...), 
    anatom_site: str = Form(...),
    needs_heatmap: bool = Form(False)
):
    
    image_bytes = await file.read()
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    processed_image = apply_vignette(original_image)
    
    
    input_batch = apply_tta(processed_image)
    with torch.no_grad():
        logits = ai_models.lightning_model(input_batch)
        cnn_probs = F.softmax(logits, dim=1).mean(dim=0).cpu().numpy()

    
    meta_data = {'age_approx': age, 'sex': sex, 'anatom_site_general': anatom_site}
    for i, class_name in enumerate(CLASSES):
        meta_data[class_name] = cnn_probs[i]

    df_meta = pd.DataFrame([meta_data])
    df_meta = pd.get_dummies(df_meta, columns=['sex', 'anatom_site_general'])
    for col in ai_models.feature_columns:
        if col not in df_meta.columns: df_meta[col] = 0
    df_meta = df_meta[ai_models.feature_columns]

    final_probs = ai_models.xgb_model.predict_proba(df_meta)[0]
    top_idx = np.argmax(final_probs)
    cnn_top_idx = np.argmax(cnn_probs)

    
    malignant_indices = [0, 2, 3, 7]
    benign_indices = [1, 4, 5, 6]
    prob_mel = final_probs[0]

    cnn_top_idx = int(np.argmax(cnn_probs))
    cnn_max_conf = float(cnn_probs[cnn_top_idx])

    
    final_diagnosis = CLASSES[top_idx].upper()
   

    if top_idx in malignant_indices:
        
        is_risky = True
    elif cnn_top_idx in malignant_indices and cnn_max_conf > 0.40:
        
        is_risky = True
        final_diagnosis = f"{CLASSES[cnn_top_idx].upper()} (VISUAL ALERT)"
    elif prob_mel > 0.45:
        
        is_risky = True
    else:
        is_risky = False

    encoded_heatmap = ""
    if needs_heatmap:
        encoded_heatmap = generate_heatmap(
            ai_models.lightning_model, 
            processed_image, 
            original_image
        )
    
    top_idx = int(np.argmax(final_probs))       
    cnn_top_idx = int(np.argmax(cnn_probs))     
    
    
    has_conflict = bool(top_idx != cnn_top_idx) 

    
    hybrid_all_probs = {CLASSES[i].upper(): float(final_probs[i]) for i in range(len(CLASSES))}
    cnn_all_probs = {CLASSES[i].upper(): float(cnn_probs[i]) for i in range(len(CLASSES))}

    return {
        "prediction": "Risky" if is_risky else "Benign",
        "diagnosis": CLASSES[top_idx].upper(),
        "confidence": float(final_probs[top_idx]),
        "all_probabilities": dict(sorted(hybrid_all_probs.items(), key=lambda x: x[1], reverse=True)),
        "heatmap_base64": encoded_heatmap,
        "metadata_used": {"age": age, "sex": sex, "site": anatom_site},
        "debug": {
            "cnn_diagnosis": CLASSES[cnn_top_idx].upper(),
            "cnn_confidence": float(cnn_probs[cnn_top_idx]),
            "cnn_all_probabilities": dict(sorted(cnn_all_probs.items(), key=lambda x: x[1], reverse=True)),
            "conflict": has_conflict  
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)