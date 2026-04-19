from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import shap
import os
import datetime
import json
import time

from monitor import record_prediction, start_metrics_server

app = FastAPI(title="Loan Approval Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Start Prometheus metrics server
try:
    start_metrics_server(8001)
except Exception as e:
    print(f"Metrics server already running or failed to start: {e}")

class LoanRequest(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: str
    Property_Area: str

@app.on_event("startup")
async def startup_event():
    global model, model_columns, categorical_cols, df_orig
    try:
        model = joblib.load('best_model.pkl')
        model_columns = joblib.load('model_columns.pkl')
        categorical_cols = joblib.load('categorical_cols.pkl')
        df_orig = pd.read_csv('loan_data.csv')
    except Exception as e:
        print("Error loading models or dataset:", e)
        model = None



@app.post("/predict")
def predict_loan(req: LoanRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded. Train the model first.")
    
    start_time = time.time()
    
    input_data = pd.DataFrame([{
        'Gender': req.Gender,
        'Married': req.Married,
        'Dependents': req.Dependents,
        'Education': req.Education,
        'Self_Employed': req.Self_Employed,
        'ApplicantIncome': req.ApplicantIncome,
        'CoapplicantIncome': req.CoapplicantIncome,
        'LoanAmount': req.LoanAmount / 1000.0,
        'Loan_Amount_Term': req.Loan_Amount_Term,
        'Credit_History': 1.0 if req.Credit_History == "Yes" else 0.0,
        'Property_Area': req.Property_Area
    }])

    # Feature Engineering
    input_data['TotalIncome'] = input_data['ApplicantIncome'] + input_data['CoapplicantIncome']
    input_data['EMI'] = input_data['LoanAmount'] / input_data['Loan_Amount_Term']
    input_data['BalanceIncome'] = input_data['TotalIncome'] - input_data['EMI']

    # One-Hot Encoding
    input_encoded = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)
    
    # Ensure all columns exist
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
            
    input_encoded = input_encoded[model_columns]

    # Predict
    prediction = int(model.predict(input_encoded)[0])
    probability = model.predict_proba(input_encoded)[0]
    confidence = float(max(probability))

    # Log prediction
    result_str = 'Approved' if prediction == 1 else 'Rejected'
    latency = time.time() - start_time
    record_prediction(result_str.lower(), confidence, latency)

    log_entry = {
        'timestamp': str(datetime.datetime.now()),
        'applicant_income': req.ApplicantIncome,
        'coapplicant_income': req.CoapplicantIncome,
        'loan_amount': req.LoanAmount,
        'credit_history': req.Credit_History,
        'prediction': result_str,
        'confidence': confidence
    }

    log_file = 'logs/predictions.csv'
    os.makedirs('logs', exist_ok=True)
    log_df = pd.DataFrame([log_entry])
    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file, index=False)

    return {
        "prediction": prediction,
        "status": result_str,
        "confidence": confidence,
        "details": {
            "TotalIncome": float(input_data['TotalIncome'][0]),
            "EMI": float(req.LoanAmount / req.Loan_Amount_Term),
            "BalanceIncome": float(input_data['TotalIncome'][0] - (req.LoanAmount / req.Loan_Amount_Term))
        }
    }

@app.get("/metrics")
def get_metrics():
    if not os.path.exists('metrics.json'):
        raise HTTPException(status_code=404, detail="Metrics not found. Run train.py first.")
    with open('metrics.json', 'r') as f:
        metrics = json.load(f)
    return metrics

@app.get("/data/summary")
def get_data_summary():
    try:
        total_records = len(df_orig)
        total_features = len(df_orig.columns)
        approved_count = int((df_orig['Loan_Status'] == 1).sum()) if 'Loan_Status' in df_orig.columns else int((df_orig.get('Loan_Status', df_orig.get('Loan_Status_Y')) == 1).sum() if 'Loan_Status_Y' in df_orig.columns else 0)
        
        # sample records
        sample_json_str = df_orig.head(10).to_json(orient="records")
        sample = json.loads(sample_json_str)
        return {
            "totalRecords": total_records,
            "totalFeatures": total_features,
            "approvedCount": approved_count,
            "sample": sample
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/shap")
def get_shap_importance():
    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    try:
        # We perform simple feature importance from SHAP on a small sample to avoid long delays
        sample_df = df_orig.copy().dropna()
        if 'Loan_ID' in sample_df.columns:
            sample_df = sample_df.drop('Loan_ID', axis=1)
        if 'Unnamed: 0' in sample_df.columns:
            sample_df = sample_df.drop('Unnamed: 0', axis=1)

        numerical_cols = sample_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        cat_cols = sample_df.select_dtypes(include=['object']).columns.tolist()
        if 'Loan_Status' in numerical_cols: numerical_cols.remove('Loan_Status')
        if 'Loan_Status' in cat_cols: cat_cols.remove('Loan_Status')

        sample_df['TotalIncome'] = sample_df['ApplicantIncome'] + sample_df['CoapplicantIncome']
        sample_df['EMI'] = sample_df['LoanAmount'] / sample_df['Loan_Amount_Term']
        sample_df['BalanceIncome'] = sample_df['TotalIncome'] - sample_df['EMI']

        df_enc = pd.get_dummies(sample_df, columns=cat_cols, drop_first=True)
        if 'Loan_Status' in df_enc.columns:
            X = df_enc.drop('Loan_Status', axis=1)
        else:
            X = df_enc
            
        # Ensure correct columns
        for col in model_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[model_columns].head(100) # sample for speed
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values
            
        importance = np.abs(shap_vals).mean(axis=0)
        importance = np.nan_to_num(importance, nan=0.0)
        imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
        imp_df = imp_df.sort_values('Importance', ascending=False)
        
        return {
            "features": imp_df['Feature'].tolist(),
            "importance": imp_df['Importance'].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files at the end so it doesn't override API routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")

