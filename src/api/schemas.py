from pydantic import BaseModel
from typing import Dict, Optional



class DebugInfo(BaseModel):

    mode: str
    conflict: bool

    cnn_diagnosis: Optional[str] = None
    cnn_confidence: Optional[float] = None
    cnn_all_probabilities: Optional[Dict[str, float]] = None

    message: Optional[str] = None
    low_confidence_warning: Optional[bool] = None


    cnn_contribution: Optional[Dict[str,float]] = None
class AnalysisResponse(BaseModel):
    clinical_report: str = ""
    prediction: str
    diagnosis: str
    confidence: float
    all_probabilities: Dict[str, float]
    heatmap_base64: str
    metadata_used: Dict[str, str]
    debug: DebugInfo

class FeedbackCreate(BaseModel):
    inference_id : int
    is_correct: bool
    corrected_diagnosis: Optional[str] = None
    doctor_notes : Optional[str] = None
    