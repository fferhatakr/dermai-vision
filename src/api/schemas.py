from pydantic import BaseModel
from typing import Dict

class DebugInfo(BaseModel):
    cnn_diagnosis: str
    cnn_confidence: float
    cnn_all_probabilities: Dict[str, float]
    conflict: bool

class AnalysisResponse(BaseModel):
    prediction: str
    diagnosis: str
    confidence: float
    all_probabilities: Dict[str, float]
    heatmap_base64: str
    metadata_used: Dict[str, str]
    debug: DebugInfo