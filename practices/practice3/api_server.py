from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
from schemas import PulsarFeatures, PredictionResponse

app = FastAPI(title="Pulsar Classification API")

try:
    with open('models_pack.pkl', 'rb') as f:
        data_pack = pickle.load(f)
    models = data_pack['models']
    scaler = data_pack['scaler']
except FileNotFoundError:
    raise RuntimeError("Файл models_pack.pkl не найден. Сначала запустите save_model.py")


@app.get("/")
def read_root():
    return {"message": "API для классификации пульсаров запущено", "available_models": list(models.keys())}


@app.post("/predict/{model_name}", response_model=PredictionResponse)
def predict(model_name: str, features: PulsarFeatures):
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Модель не найдена")

    input_data = np.array([[
        features.mean_profile, features.sd_profile, features.kurt_profile, features.skew_profile,
        features.mean_dmsnr, features.sd_dmsnr, features.kurt_dmsnr, features.skew_dmsnr
    ]])

    input_scaled = scaler.transform(input_data)

    model = models[model_name]
    prediction = int(model.predict(input_scaled)[0])

    probability = 0.0
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_scaled)[0][1])

    return {
        "model_name": model_name,
        "prediction": prediction,
        "probability": probability
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)