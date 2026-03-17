from fastapi import APIRouter, Depends, UploadFile, File, Form
from src.api.schemas import AnalysisResponse
from src.api.auth import get_current_user
from src.api.models import  CLASSES
from src.api.inference import apply_tta, apply_vignette
from src.api.gradcam import generate_heatmap
from src.api.database import get_db
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import io
import src.api.models as ai_models
from src.api import db_models
from sqlalchemy.orm import Session

router = APIRouter(tags=["analysis"])


@router.post("/analyze",response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...), 
    age: float = Form(...), 
    sex: str = Form(...), 
    anatom_site: str = Form(...),
    needs_heatmap: bool = Form(False),
    current_user: str = Depends(get_current_user),
    full_name: str = Form(...),
    db: Session = Depends(get_db)
    ):
    existing_patient = db.query(db_models.Patient).filter(
        db_models.Patient.full_name == full_name
    ).first()
    
    if existing_patient:
        patient = existing_patient
    else:
        patient = db_models.Patient(
            full_name = full_name,
            age=int(age),
            sex=sex,
            anatom_site=anatom_site
        )
        db.add(patient)
        db.commit()
        db.refresh(patient)




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


    #Find Doctor id
    db_user = db.query(db_models.User).filter(
        db_models.User.email == current_user
    ).first()

    new_analysis = db_models.Analysis(
        patient_id = patient.id,
        user_id = db_user.id,
        diagnosis = CLASSES[top_idx].upper(),
        confidence=float(final_probs[top_idx]),
        is_risky=is_risky
    )
    db.add(new_analysis)
    db.commit()

    return {
        "prediction": "Risky" if is_risky else "Benign",
        "diagnosis": CLASSES[top_idx].upper(),
        "confidence": float(final_probs[top_idx]),
        "all_probabilities": dict(sorted(hybrid_all_probs.items(), key=lambda x: x[1], reverse=True)),
        "heatmap_base64": encoded_heatmap,
        "metadata_used": {"age": str(age), "sex": sex, "site": anatom_site},
        "debug": {
            "cnn_diagnosis": CLASSES[cnn_top_idx].upper(),
            "cnn_confidence": float(cnn_probs[cnn_top_idx]),
            "cnn_all_probabilities": dict(sorted(cnn_all_probs.items(), key=lambda x: x[1], reverse=True)),
            "conflict": has_conflict  
        }
    }