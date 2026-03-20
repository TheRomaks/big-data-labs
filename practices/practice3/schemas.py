from pydantic import BaseModel

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