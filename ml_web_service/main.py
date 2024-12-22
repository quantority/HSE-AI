import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from http import HTTPStatus
from typing import Dict, List, Any
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np

app = FastAPI()


class ModelConfig(BaseModel):
    id: str
    hyperparameters: Dict[str, Any]


class FitRequest(BaseModel):
    X: List[List[float]]
    y: List[float]
    config: ModelConfig


class FitResponse(BaseModel):
    message: str


class LoadRequest(BaseModel):
    id: str


class LoadResponse(BaseModel):
    message: str


class PredictRequest(BaseModel):
    id: str
    X: List[List[float]]


class PredictionResponse(BaseModel):
    id: str
    predictions: List[float]


class ModelListResponse(BaseModel):
    id: str


class RemoveResponse(BaseModel):
    message: str


# Словарь для хранения моделей
models: Dict[str, Dict] = {}


@app.post("/fit", response_model=List[FitResponse], status_code=HTTPStatus.CREATED)
async def fit(request: List[FitRequest]):
    responses = []
    
    for item in request:
        model = LinearRegression(**item.config.hyperparameters)
        X_np = np.array(item.X)
        y_np = np.array(item.y)
        model.fit(X_np, y_np)
        
        models[item.config.id] = {
            'model': model,
            'config': item.config.dict()
        }
        
        # Используем точно такой же формат сообщения как в спецификации
        responses.append(FitResponse(message=f"Model 'Linear_{item.config.id}' trained and saved"))
    
    return responses


@app.post("/load", response_model=List[LoadResponse])
async def load(request: LoadRequest):
    if request.id not in models:
        return [LoadResponse(message=f"Model '{request.id}' not found")]
    return [LoadResponse(message=f"Model '{request.id}' loaded")]


@app.post("/predict", response_model=List[PredictionResponse])
async def predict(request: PredictRequest):
    if request.id not in models:
        return [PredictionResponse(id=request.id, predictions=[])]
    
    model = models[request.id]['model']
    X_np = np.array(request.X)
    predictions = model.predict(X_np)
    
    return [PredictionResponse(id=request.id, predictions=predictions.tolist())]


@app.get("/list_models", response_model=List[ModelListResponse])
async def list_models():
    return [ModelListResponse(id=model_id) for model_id in models.keys()]


@app.delete("/remove_all", response_model=List[RemoveResponse])
async def remove_all():
    model_ids = list(models.keys())
    models.clear()
    return [RemoveResponse(message=f"Model removed")] if model_ids else [RemoveResponse(message="No models to remove")]


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)