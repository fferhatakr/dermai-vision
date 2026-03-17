from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from src.api.database import get_db
from src.api import db_models
from src.api.auth import get_current_user
from pydantic import BaseModel

router = APIRouter(prefix="/patients", tags=["patients"])

class PatientCreate(BaseModel):
    full_name: str
    age: int
    sex: str
    anatom_site: str

@router.post("/add")
def add_patient(
    patient: PatientCreate,
    db: Session = Depends(get_db),
    current_user:str = Depends(get_current_user)

):
    new_patient = db_models.Patient(
        full_name = patient.full_name,
        age=patient.age,
        sex=patient.sex,
        anatom_site = patient.anatom_site
    )
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)

    return {"message": "Patient added","patient_id": new_patient.id}


@router.get("/list")
def list_patients(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    patients = db.query(db_models.Patient).all()
    
    return [{"id": p.id, "full_name": p.full_name}for p in patients]


@router.get("/{patient_id}/analyses")
def get_patient_analyses(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    patient = db.query(db_models.Patient).filter(
        db_models.Patient.id == patient_id
    ).first()

    if not patient:
        raise HTTPException(
            status_code= status.HTTP_404_NOT_FOUND,
            detail = "Patient not found"
        )
    analyses = db.query(db_models.Analysis).filter(
        db_models.Analysis.patient_id == patient_id
    ).all()

    return {
        "patient": {
            "id": patient.id,
            "full_name": patient.full_name,
            "age": patient.age,
            "sex": patient.sex,
            "anatom_site": patient.anatom_site
        },
        "analyses": [
            {
                "id": a.id,
                "diagnosis": a.diagnosis,
                "confidence": a.confidence,
                "is_risky": a.is_risky,
                "created_at": str(a.created_at)
            }
            for a in analyses
        ],
        "total_analyses": len(analyses)
    }
