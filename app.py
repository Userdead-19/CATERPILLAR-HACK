from flask import Flask, jsonify
import paho.mqtt.client as mqtt
import threading
import json
import time
from pymongo import MongoClient
from ml_model import detect_anomalies_batch  # Updated model
import queue
import logging
from bson import ObjectId
from flask_cors import CORS


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
            message["_id"] = str(message["_id"])  # Convert ObjectId to string

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
    while True:
        time.sleep(10)  # 10-second batch window
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

        try:
            alerts = detect_anomalies_batch(batch)
            for alert in alerts:
                logging.warning(f"⚠️ Anomaly Detected: {alert}")
                publish_alert(alert)
        except Exception as e:
            logging.exception("Anomaly detection failed.")

        global latest_data
        latest_data = batch[-1]


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
