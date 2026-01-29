from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI(title="Credit Risk Scoring API")

model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "Credit Risk Scoring API is running"}

@app.post("/predict")
def predict(data: dict):
    features = np.array([[
        data["ApplicantIncome"],
        data["CoapplicantIncome"],
        data["LoanAmount"],
        data["Loan_Amount_Term"],
        data["Credit_History"]
    ]])

    features = scaler.transform(features)
    prob = model.predict_proba(features)[0][1]

    risk = "HIGH" if prob > 0.6 else "MEDIUM" if prob > 0.3 else "LOW"

    return {
        "default_probability": round(float(prob), 3),
        "risk_level": risk
    }
