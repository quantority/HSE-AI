from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import io
import pandas as pd

app = FastAPI()

with open('lr-pipeline.pkl', 'rb') as f:
    model = pickle.load(f)


class Item(BaseModel):
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
async def predict_item(item: Item) -> float:
    try:
        X_item = [[
            item.year,
            item.km_driven,
            float(item.mileage.split()[0]),
            float(item.engine.split()[0]),
            float(item.max_power.split()[0]),
            item.seats
        ]]

        # Делаем предсказание
        y_pred = model.predict(X_item)[0]
        return float(y_pred)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Не удалось сделать предсказание на объекте {str(e)}"
            )


@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)):
    try:
        # Читаем CSV файл
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Проверяем наличие всех необходимых колонок
        required_fields = Item.__annotations__.keys()
        missing_fields = [
            field for field in required_fields if field not in df.columns
            ]
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Отсутствуют обязательные поля: {missing_fields}"
            )

        X = df[
            [
                'year',
                'km_driven',
                'mileage',
                'engine',
                'max_power',
                'seats'
                ]
                ]

        df['predictions'] = model.predict(X)
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return {"predicted_csv": output.getvalue()}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обработки CSV: {str(e)}"
            )
