from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import joblib
import numpy as np

app = FastAPI()

model = joblib.load('./artifacts/random_forest_model.pkl')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="O arquivo deve ser um CSV.")

    df = pd.read_csv(file.file)

    columns = [
        'nr_atividades_mapeadas',
        'nr_questionarios_finalizados',
        'nr_intervalos_uso',
        'vl_medio_atividade_diaria',
        'nr_interacoes_usuario',
        'vl_engajamento_notas',
        'vl_media_notas'
    ]  
    
    if not all(column in df.columns for column in columns):
        raise HTTPException(status_code=400, detail=f"CSV deve conter as colunas: {columns}")

    predictions = model.predict(df[columns])

    return { "previsoes": predictions.tolist() }
