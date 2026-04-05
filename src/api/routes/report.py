from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.api.auth import get_current_user
from src.api.llm_report import generate_clinical_report

router = APIRouter(tags=["report"])

class ReportRequest(BaseModel):
    diagnosis: str           
    confidence: float        
    all_probabilities: dict  
    is_risky: bool          
    age: float               
    sex: str               
    anatom_site: str         

class ReportResponse(BaseModel):
    clinical_report: str

@router.post("/report", response_model=ReportResponse)
async def generate_Report(
    request: ReportRequest,
    current_user: str = Depends(get_current_user)
):
    report = await generate_clinical_report(
        diagnosis=request.diagnosis,
        confidence=request.confidence,
        all_probs=request.all_probabilities,
        is_risky=request.is_risky,
        age=request.age,
        sex=request.sex,
        anatom_site=request.anatom_site
    )

    return {"clinical_report": report}