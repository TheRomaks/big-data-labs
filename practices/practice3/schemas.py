from pydantic import BaseModel
from typing import List, Dict, Any

class PulsarFeatures(BaseModel):
    mean_profile: float
    sd_profile: float
    kurt_profile: float
    skew_profile: float
    mean_dmsnr: float
    sd_dmsnr: float
    kurt_dmsnr: float
    skew_dmsnr: float

class PredictionResponse(BaseModel):
    model_name: str
    prediction: int
    probability: float = None

class AutoMLResults(BaseModel):
    best_model_name: str
    best_config: Dict[str, Any]
    best_f1: float
    comparison_table: List[Dict[str, Any]]