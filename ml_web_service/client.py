import asyncio
import aiohttp
import numpy as np
from sklearn.datasets import make_regression
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"

async def train_model(session, model_id: str, X, y, hyperparameters=None):
    if hyperparameters is None:
        hyperparameters = {'fit_intercept': True}
    
    async with session.post(
        f"{BASE_URL}/fit",
        json=[{
            "X": X.tolist(),
            "y": y.tolist(),
            "config": {
                "id": model_id,
                "ml_model_type": "linear",
                "hyperparameters": hyperparameters
            }
        }]
    ) as response:
        if response.status == 201:
            result = await response.json()
            logger.info(f"Model training result: {result}")
        else:
            error_text = await response.text()
            logger.error(f"Error training model: {error_text}")
            raise Exception(error_text)

async def load_model(session, model_id: str):
    async with session.post(
        f"{BASE_URL}/load",
        json={"id": model_id}
    ) as response:
        if response.status == 200:
            result = await response.json()
            logger.info(f"Model loading result: {result}")
        else:
            error_text = await response.text()
            logger.error(f"Error loading model: {error_text}")
            raise Exception(error_text)

async def predict(session, model_id: str, X):
    async with session.post(
        f"{BASE_URL}/predict",
        json={
            "id": model_id,
            "X": X.tolist()
        }
    ) as response:
        if response.status == 200:
            result = await response.json()
            return result[0]["predictions"]
        else:
            error_text = await response.text()
            logger.error(f"Error making prediction: {error_text}")
            raise Exception(error_text)

async def list_models(session):
    async with session.get(f"{BASE_URL}/list_models") as response:
        if response.status == 200:
            result = await response.json()
            logger.info(f"Available models: {result}")
            return [item["id"] for item in result]
        else:
            error_text = await response.text()
            logger.error(f"Error listing models: {error_text}")
            raise Exception(error_text)

async def remove_all_models(session):
    async with session.delete(f"{BASE_URL}/remove_all") as response:
        if response.status == 200:
            result = await response.json()
            logger.info(f"Remove all models result: {result}")
        else:
            error_text = await response.text()
            logger.error(f"Error removing models: {error_text}")
            raise Exception(error_text)

async def main():
    async with aiohttp.ClientSession() as session:
        try:
            X, y = make_regression(n_samples=1000, n_features=20, noise=0.1)
            
            await remove_all_models(session)
            
            model_id = "linear_model_1"
            logger.info(f"Training model {model_id}...")
            await train_model(
                session,
                model_id,
                X,
                y,
                {"fit_intercept": True}
            )
            
            models = await list_models(session)
            logger.info(f"Available models: {models}")
            
            await load_model(session, model_id)
            
            prediction_tasks = []
            for i in range(5):
                X_subset = X[i*10:(i+1)*10]
                prediction_tasks.append(predict(session, model_id, X_subset))
            
            predictions = await asyncio.gather(*prediction_tasks)
            logger.info(f"Received {len(predictions)} predictions")
            
        except Exception as e:
            logger.error(f"Error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())