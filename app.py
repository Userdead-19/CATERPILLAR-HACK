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
    task_time_pipeline = joblib.load("task_time_prediction.pkl")
    task_time_features = joblib.load("task_time_features.pkl")
    logging.info("Task time estimation model loaded successfully.")
    logging.info(f"Expected features: {list(task_time_features)}")
except Exception as e:
    logging.exception("Failed to load task time model.")
    task_time_pipeline = None
    task_time_features = None


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


def predict_task_time(input_features):
    """
    Predict task completion time using the trained pipeline.

    Args:
        input_features (dict): Dictionary containing all required features

    Returns:
        float: Predicted task completion time in minutes
    """
    if not task_time_pipeline:
        logging.error("Task time model not loaded.")
        return None

    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_features])

        # Make prediction using the pipeline (handles preprocessing automatically)
        prediction = task_time_pipeline.predict(input_df)

        logging.info(f"Task time prediction: {prediction[0]:.2f} minutes")
        return float(prediction[0])

    except Exception as e:
        logging.exception(f"Error in task time prediction: {str(e)}")
        return None


def validate_prediction_input(features):
    """
    Validate that all required features are present in the input.

    Args:
        features (dict): Input features dictionary

    Returns:
        tuple: (is_valid, error_message)
    """
    required_features = [
        "Engine_Hours",
        "Fuel_Used_L",
        "Load_Cycles",
        "Idling_Time_min",
        "Seatbelt_Status",
        "Safety_Alert",
        "Operator_ID",
        "Weather_Temp_C",
        "Weather_Rainfall_mm",
        "Weather_Wind_kmph",
        "Terrain",
        "Task_Type",
    ]

    missing_features = [f for f in required_features if f not in features]

    if missing_features:
        return False, f"Missing required features: {missing_features}"

    # Validate categorical features
    valid_terrains = ["Flat", "Muddy", "Rocky"]
    valid_task_types = ["Digging", "Grading", "Loading", "Transport"]

    if features.get("Terrain") not in valid_terrains:
        logging.warning(
            f"Unknown terrain '{features.get('Terrain')}' - will be handled as unknown category"
        )

    if features.get("Task_Type") not in valid_task_types:
        logging.warning(
            f"Unknown task type '{features.get('Task_Type')}' - will be handled as unknown category"
        )

    return True, None


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
    """
    API endpoint for task time prediction.

    Expected JSON input:
    {
        "Engine_Hours": 5.0,
        "Fuel_Used_L": 45.0,
        "Load_Cycles": 100,
        "Idling_Time_min": 30,
        "Seatbelt_Status": 1,
        "Safety_Alert": 0,
        "Operator_ID": 15,
        "Weather_Temp_C": 25.0,
        "Weather_Rainfall_mm": 10.0,
        "Weather_Wind_kmph": 15.0,
        "Terrain": "Rocky",
        "Task_Type": "Loading"
    }
    """
    logging.info("POST /predict-task-time called")

    try:
        # Get JSON data from request
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400

        features = request.json
        logging.info(f"Received features: {features}")

        # Validate input
        is_valid, error_message = validate_prediction_input(features)
        if not is_valid:
            return jsonify({"error": error_message}), 400

        # Make prediction
        prediction = predict_task_time(features)

        if prediction is None:
            return jsonify({"error": "Prediction failed"}), 500

        response = {
            "estimated_time_minutes": round(prediction, 2),
            "estimated_time_hours": round(prediction / 60, 2),
            "input_features": features,
        }

        logging.info(f"Prediction response: {response}")
        return jsonify(response)

    except Exception as e:
        logging.exception("Task time prediction failed.")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/model-info", methods=["GET"])
def get_model_info():
    """
    Get information about the loaded models and expected features.
    """
    logging.info("GET /model-info called")

    info = {
        "task_time_model_loaded": task_time_pipeline is not None,
        "anomaly_model_loaded": anomaly_model is not None,
        "required_features": [
            "Engine_Hours",
            "Fuel_Used_L",
            "Load_Cycles",
            "Idling_Time_min",
            "Seatbelt_Status",
            "Safety_Alert",
            "Operator_ID",
            "Weather_Temp_C",
            "Weather_Rainfall_mm",
            "Weather_Wind_kmph",
            "Terrain",
            "Task_Type",
        ],
        "valid_terrains": ["Flat", "Muddy", "Rocky"],
        "valid_task_types": ["Digging", "Grading", "Loading", "Transport"],
    }

    if task_time_features is not None:
        info["processed_features"] = list(task_time_features)

    return jsonify(info)


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
