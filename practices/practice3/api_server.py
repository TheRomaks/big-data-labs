from fastapi import FastAPI, HTTPException
import pickle
import numpy as np

from flaml import AutoML
from schemas import PulsarFeatures, PredictionResponse, AutoMLResults
import os

app = FastAPI(title="Pulsar Classification API with AutoML")

MODELS_PATH = 'models_pack.pkl'
AUTOML_MODEL_PATH = 'automl_best.pkl'


def load_resources():
    try:
        with open(MODELS_PATH, 'rb') as f:
            data = pickle.load(f)
        return data['models'], data['scaler'], data.get('X_train'), data.get('y_train')
    except FileNotFoundError:
        return {}, None, None, None


models, scaler, X_train_raw, y_train_raw = load_resources()


@app.get("/")
def read_root():
    return {
        "message": "API готово",
        "manual_models": list(models.keys()),
        "automl_ready": os.path.exists(AUTOML_MODEL_PATH)
    }


@app.post("/train/automl", response_model=AutoMLResults)
def train_automl():
    if X_train_raw is None:
        raise HTTPException(status_code=400, detail="Данные для обучения не найдены")

    automl = AutoML()
    settings = {
        "time_budget": 30,
        "metric": 'f1',
        "task": 'classification',
        "log_file_name": 'pulsar_automl.log',
    }

    automl.fit(X_train=X_train_raw, y_train=y_train_raw, **settings)

    with open(AUTOML_MODEL_PATH, 'wb') as f:
        pickle.dump(automl, f)

    return {
        "best_model_name": automl.best_estimator,
        "best_config": automl.best_config,
        "best_f1": automl.best_loss,
        "comparison_table": [{"model": automl.best_estimator, "f1": 1 - automl.best_loss}]
    }


@app.post("/predict/{model_type}/{model_name}", response_model=PredictionResponse)
def predict(model_type: str, model_name: str, features: PulsarFeatures):
    input_data = np.array([[
        features.mean_profile, features.sd_profile, features.kurt_profile, features.skew_profile,
        features.mean_dmsnr, features.sd_dmsnr, features.kurt_dmsnr, features.skew_dmsnr
    ]])
    input_scaled = scaler.transform(input_data)

    if model_type == "manual":
        if model_name not in models:
            raise HTTPException(status_code=404, detail="Ручная модель не найдена")
        model = models[model_name]
    else:
        if not os.path.exists(AUTOML_MODEL_PATH):
            raise HTTPException(status_code=404, detail="AutoML модель не обучена")
        with open(AUTOML_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

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