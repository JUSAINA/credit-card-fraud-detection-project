from fastapi import FastAPI
from schema import Transaction
from predict import predict_transaction

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Fraud Detection using Machine Learning",
    version="1.0.0"
)


@app.get("/")
def home():

    return {
        "message": "Credit Card Fraud Detection API"
    }


@app.post("/predict")
def predict(transaction: Transaction):

    prediction, probability = predict_transaction(
        transaction.dict()
    )

    return {

        "prediction": int(prediction),

        "probability": float(probability),

        "label":
            "Fraud"
            if prediction == 1
            else "Normal"
    }