import joblib
import numpy as np
import pandas as pd

model = joblib.load("../models/final_model.pkl")
scaler = joblib.load("../models/scaler.pkl")


def preprocess(data):

    df = pd.DataFrame([data])

    df["Log_Amount"] = np.log1p(df["Amount"])

    df["Hour"] = (df["Time"] // 3600).astype(int)

    df.drop(columns=["Amount"], inplace=True)

    df = df[
        [
            "Time",
            "V1","V2","V3","V4","V5","V6",
            "V7","V8","V9","V10","V11","V12",
            "V13","V14","V15","V16","V17",
            "V18","V19","V20","V21","V22",
            "V23","V24","V25","V26","V27",
            "V28","Log_Amount","Hour"
        ]
    ]

    df_scaled = scaler.transform(df)

    return df_scaled


def predict_transaction(data):

    processed = preprocess(data)

    prediction = model.predict(processed)[0]

    probability = model.predict_proba(processed)[0][1]

    return prediction, probability
