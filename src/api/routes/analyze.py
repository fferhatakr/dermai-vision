from fastapi import APIRouter, Depends, UploadFile, File, Form
from src.api.schemas import AnalysisResponse
from src.api.auth import get_current_user
from src.api.models import CLASSES
from src.api.inference import run_hybrid_inference, apply_vignette , run_quick_scan, run_standard_analysis
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


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    age: float = Form(...),
    sex: str = Form(...),
    anatom_site: str = Form(...),
    needs_heatmap: bool = Form(False),
    current_user: str = Depends(get_current_user),
    full_name: str = Form(...),
    db: Session = Depends(get_db),
    analysis_mode: str = Form(default="detailed")
):
    existing_patient = db.query(db_models.Patient).filter(
        db_models.Patient.full_name == full_name
    ).first()

    if existing_patient:
        patient = existing_patient
    else:
        patient = db_models.Patient(
            full_name=full_name,
            age=int(age),
            sex=sex,
            anatom_site=anatom_site
        )
        db.add(patient)
        db.commit()
        db.refresh(patient)

    image_bytes = await file.read()
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    cv_image = np.array(original_image)
    cv_image = cv_image[:, :, ::-1].copy()

    img_w, img_h = original_image.size

    
    margin = 0.20
    center_crop = original_image.crop((
        int(img_w * margin), int(img_h * margin),
        int(img_w * (1 - margin)), int(img_h * (1 - margin))
    ))

    
    results = ai_models.yolo_model.predict(cv_image, conf=0.25, verbose=False)
    cropped_image = center_crop  

    for r in results:
        if len(r.boxes) > 0:
            b = r.boxes.xyxy[0].cpu().numpy().astype(int)
            box_w = b[2] - b[0]
            box_h = b[3] - b[1]

            
            if box_w < img_w * 0.6 and box_h < img_h * 0.6:
                cropped_image = original_image.crop((b[0], b[1], b[2], b[3]))
                print("YOLO crop used")
            else:
                print("YOLO too wide — center crop used")
            break
    else:
        print("YOLO no detection — center crop fallback")

    processed_image = apply_vignette(cropped_image)

    if analysis_mode == "quick":

        result = run_quick_scan(processed_image)

        is_risky = result["is_risky"]
        final_diagnosis = result["top_class"]
        confidence = result["confidence"]

        db_user = db.query(db_models.User).filter(
            db_models.User.email == current_user
        ).first()
        new_analysis = db_models.Analysis(
            patient_id=patient.id,
            user_id=db_user.id,
            diagnosis=final_diagnosis,
            confidence=confidence,
            is_risky=is_risky
        )
        db.add(new_analysis)
        db.commit()

        return {
            "prediction": "Risky" if is_risky else "Benign",
            "diagnosis": final_diagnosis,
            "confidence": confidence,
            "all_probabilities": {}, 
            "heatmap_base64": "",     
            "metadata_used": {},      
            "debug": {
                "mode": "quick_scan",
                "low_confidence_warning": result["low_confidence_warning"],
                "message": result["message"],
                "conflict": False
            }
        }
    elif analysis_mode == "standard":
        result = run_standard_analysis(processed_image, age, sex, anatom_site)
        final_probs = list(result["all_probabilities"].values())
        top_idx = int(np.argmax(final_probs))
        is_risky = top_idx in [0, 2, 3, 7]
        confidence = result["confidence"]

        db_user = db.query(db_models.User).filter(
            db_models.User.email == current_user
        ).first()
        new_analysis = db_models.Analysis(
            patient_id=patient.id,
            user_id=db_user.id,
            diagnosis=result["prediction"],
            confidence=confidence,
            is_risky=is_risky
        )
        db.add(new_analysis)
        db.commit()

        return {
            "prediction": "Risky" if is_risky else "Benign",
            "diagnosis": result["prediction"],
            "confidence": confidence,
            "all_probabilities": result["all_probabilities"],
            "heatmap_base64": "",  
            "metadata_used": {"age": str(age), "sex": sex, "site": anatom_site},
            "debug": {
                "mode": "standard_analysis",
                "cnn_contribution": result["cnn_contribution"],
                "low_confidence_warning": result["low_confidence_warning"],
                "conflict": False
            }
        }
    else:
    

    
        inference_result = run_hybrid_inference(
            processed_image, 
            age, 
            sex, 
            anatom_site, 
            ai_models.lightning_model
        )

    
        final_probs = list(inference_result["all_probabilities"].values())
        cnn_probs = list(inference_result["cnn_contribution"].values())
        
        top_idx = int(np.argmax(final_probs))
        cnn_top_idx = int(np.argmax(cnn_probs))
        cnn_max_conf = float(cnn_probs[cnn_top_idx])
        prob_mel = float(inference_result["all_probabilities"]["MEL"])

        final_diagnosis = CLASSES[top_idx].upper()

    
        malignant_indices = [0, 2, 3, 7]
        
        if top_idx in malignant_indices:
            is_risky = True
        elif cnn_top_idx in malignant_indices and cnn_max_conf > 0.40:
            is_risky = True 
            final_diagnosis = f"{CLASSES[cnn_top_idx].upper()} (VISUAL ALERT)"
        elif prob_mel > 0.11: 
            is_risky = True
            final_diagnosis = "MELANOMA RISK (LOW THRESHOLD)"
        else:
            is_risky = False

        encoded_heatmap = ""
        if needs_heatmap:
            encoded_heatmap = generate_heatmap(
                ai_models.lightning_model,
                processed_image,
                cropped_image
            )

        has_conflict = bool(top_idx != cnn_top_idx)

        hybrid_all_probs = {CLASSES[i].upper(): float(final_probs[i]) for i in range(len(CLASSES))}
        cnn_all_probs = {CLASSES[i].upper(): float(cnn_probs[i]) for i in range(len(CLASSES))}

        db_user = db.query(db_models.User).filter(
            db_models.User.email == current_user
        ).first()

        new_analysis = db_models.Analysis(
            patient_id=patient.id,
            user_id=db_user.id,
            diagnosis=CLASSES[top_idx].upper(),
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
                "mode": "detailed_analysis",
                "cnn_diagnosis": CLASSES[cnn_top_idx].upper(),
                "cnn_confidence": float(cnn_probs[cnn_top_idx]),
                "cnn_all_probabilities": dict(sorted(cnn_all_probs.items(), key=lambda x: x[1], reverse=True)),
                "conflict": has_conflict
            }
        }