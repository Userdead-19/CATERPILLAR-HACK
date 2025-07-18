from flask import Flask, jsonify, request
import paho.mqtt.client as mqtt
import threading
import json
import time
from pymongo import MongoClient
import queue
import logging
from bson import ObjectId
from flask_cors import CORS
import pandas as pd
import tensorflow as tf
import joblib
import numpy as np

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("iot_app.log"), logging.StreamHandler()],
)

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

# --- MongoDB Setup ---
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["iot_db"]
collection = db["machine_logs"]

MQTT_BROKER = "test.mosquitto.org"
MQTT_TOPIC = "iot/machine/data"
NOTIFY_TOPIC = "iot/machine/alerts"

# --- Queue and Data Store ---
message_queue = queue.Queue()
latest_data = {}

# --- Load Anomaly Detection Model ---
try:
    model_package = joblib.load("multi_anomaly_nn_meta.pkl")
    anomaly_model = tf.keras.models.load_model("multi_anomaly_nn_model.h5")
    feature_columns = model_package["features"]
    anomaly_labels = model_package["labels"]
    logging.info("Anomaly detection model loaded successfully.")
except Exception as e:
    logging.exception("Failed to load anomaly detection model.")
    anomaly_model = None
    feature_columns = []
    anomaly_labels = []

# --- Load Task Time Estimation Model ---
try:
    task_time_model = joblib.load("task_time_prediction.pkl")
    task_time_features = joblib.load("task_time_features.pkl")
    logging.info("Task time estimation model loaded successfully.")
except Exception as e:
    logging.exception("Failed to load task time model.")
    task_time_model = None


# --- MQTT Callbacks ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logging.info("Connected to MQTT broker.")
    else:
        logging.error(f"Failed to connect, return code {rc}")
    client.subscribe(MQTT_TOPIC)
    logging.info(f"Subscribed to topic: {MQTT_TOPIC}")


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        message_queue.put(payload)
        logging.info(f"Received message: {payload}")
    except Exception as e:
        logging.exception("Failed to decode MQTT message.")


def publish_alert(message):
    try:
        if "_id" in message:
            message["_id"] = str(message["_id"])

        alert_client = mqtt.Client()
        alert_client.connect(MQTT_BROKER, 1883, 60)
        alert_client.loop_start()
        alert_client.publish(NOTIFY_TOPIC, json.dumps(message))
        alert_client.loop_stop()

        logging.warning(f"⚠️ Published anomaly alert: {message}")
    except Exception as e:
        logging.exception("Failed to publish alert.")


# --- Batch Processor ---
def batch_processor():
    global latest_data

    while True:
        time.sleep(10)
        batch = []

        while not message_queue.empty():
            batch.append(message_queue.get())

        if not batch:
            continue

        try:
            collection.insert_many(batch)
            logging.info(f"Inserted batch of {len(batch)} logs into MongoDB.")
        except Exception as e:
            logging.exception("Failed to insert batch to MongoDB.")

        if anomaly_model:
            try:
                alerts = detect_anomalies_batch(batch)
                for alert in alerts:
                    logging.warning(f"⚠️ Anomaly Detected: {alert}")
                    publish_alert(alert)
            except Exception as e:
                logging.exception("Anomaly detection failed.")

        latest_data = batch[-1]


def detect_anomalies_batch(batch):
    df = pd.DataFrame(batch)
    input_data = df[feature_columns].values
    predictions = anomaly_model.predict(input_data)
    predictions = (predictions >= 0.5).astype(int)

    alerts = []
    for idx, record in enumerate(batch):
        pred = predictions[idx].tolist()
        anomaly_flags = dict(zip(anomaly_labels, map(bool, pred)))
        if any(anomaly_flags.values()):
            record.update(anomaly_flags)
            alerts.append(record)
    return alerts


def predict_task_time(features):
    if task_time_model:
        input_df = pd.DataFrame([features])[task_time_features]
        prediction = task_time_model.predict(input_df)
        return float(prediction[0])
    else:
        logging.error("Task time model not loaded.")
        return None


def convert_objectid(data):
    if isinstance(data, list):
        return [convert_objectid(item) for item in data]
    elif isinstance(data, dict):
        return {k: str(v) if isinstance(v, ObjectId) else v for k, v in data.items()}
    return data


@app.route("/latest", methods=["GET"])
def get_latest():
    logging.info("GET /latest called")
    safe_data = convert_objectid(latest_data)
    return jsonify(safe_data)


@app.route("/all", methods=["GET"])
def get_all():
    logging.info("GET /all called")
    try:
        all_data = list(collection.find({}, {"_id": 0}))
        return jsonify(all_data)
    except Exception as e:
        logging.exception("Failed to retrieve all data from MongoDB.")
        return jsonify({"error": "Could not fetch data"}), 500


@app.route("/predict-task-time", methods=["POST"])
def predict_task_time_api():
    logging.info("POST /predict-task-time called")
    try:
        features = request.json
        prediction = predict_task_time(features)
        return jsonify({"estimated_time": prediction})
    except Exception as e:
        logging.exception("Task time prediction failed.")
        return jsonify({"error": "Prediction failed."}), 500


# --- Start MQTT Client ---
def start_mqtt():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(MQTT_BROKER, 1883, 60)
        client.loop_forever()
    except Exception as e:
        logging.exception("MQTT connection failed.")


# --- Threads ---
threading.Thread(target=start_mqtt, daemon=True).start()
threading.Thread(target=batch_processor, daemon=True).start()

if __name__ == "__main__":
    logging.info("Starting Flask app...")
    app.run(debug=True)
