import pandas as pd
import tensorflow as tf
import joblib
import numpy as np
import logging

# Load model + metadata
model = tf.keras.models.load_model("multi_anomaly_nn_model.h5")
meta = joblib.load("nn_model_meta.pkl")

feature_columns = meta["features"]
anomaly_labels = meta["labels"]


def detect_anomalies_batch(batch):
    df = pd.DataFrame(batch)
    try:
        input_data = df[feature_columns].values
    except KeyError as e:
        logging.error(f"Missing required feature columns: {e}")
        return []

    predictions = model.predict(input_data)
    predictions = (predictions >= 0.5).astype(int)  # Thresholding

    alerts = []

    for idx, record in enumerate(batch):
        pred = predictions[idx].tolist()
        anomaly_flags = dict(zip(anomaly_labels, map(bool, pred)))

        if any(anomaly_flags.values()):
            record.update(anomaly_flags)
            alerts.append(record)

        logging.info(f"Record {idx} predictions: {pred}")

    return alerts
