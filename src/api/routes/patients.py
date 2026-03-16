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
