import joblib
import numpy as np
import pandas as pd

model = joblib.load("anomaly_model.pkl")
feature_names = ["Fuel Used (L)", "Load Cycles", "Idling Time (min)", "Engine Hours"]


def detect_anomalies_batch(batch):
    results = []
    for entry in batch:
        try:
            row = pd.DataFrame(
                [[entry[f] for f in feature_names]], columns=feature_names
            )
            prediction = model.predict(row)
            if prediction[0] == -1:
                results.append(entry)
        except Exception as e:
            print("ML Error:", e)
    return results
