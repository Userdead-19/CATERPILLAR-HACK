import json
import random
import time
import paho.mqtt.client as mqtt
import pandas as pd
import numpy as np
import requests
from tensorflow.keras.models import load_model

MQTT_BROKER = "test.mosquitto.org"
DATA_TOPIC = "iot/machine/data"
ALERT_TOPIC = "iot/machine/alerts"

FLASK_API_URL = "http://localhost:5000/predict-task-time"

# Load only CNN model locally
anomaly_cnn_model = load_model("multi_anomaly_nn_model.h5")


def generate_task_time_input():
    return {
        "Engine_Hours": round(random.uniform(1523, 1535), 1),
        "Fuel_Used_L": round(random.uniform(2, 7), 1),
        "Load_Cycles": random.randint(1, 15),
        "Idling_Time_min": random.randint(5, 60),
        "Seatbelt_Status": random.choice(["Fastened", "Unfastened"]),
        "Safety_Alert": random.choice(["Yes", "No"]),
        "Operator_ID": f"OP{random.randint(1000, 1100)}",
        "Weather_Temp_C": round(random.uniform(20, 40), 1),
        "Weather_Rainfall_mm": round(random.uniform(0, 5), 1),
        "Weather_Wind_kmph": round(random.uniform(0, 20), 1),
        "Terrain": random.choice(["Rocky", "Sandy", "Muddy"]),
        "Task_Type": random.choice(["Digging", "Lifting", "Transport"]),
    }


def fetch_task_time_prediction(task_input):
    try:
        response = requests.post(FLASK_API_URL, json=task_input)
        if response.status_code == 200:
            result = response.json()
            return result.get("estimated_time")
        else:
            print("API Error:", response.status_code, response.text)
            return None
    except requests.exceptions.RequestException as e:
        print("Request Exception:", e)
        return None


def generate_anomaly_prediction_payload():
    cnn_input = pd.DataFrame(
        [
            {
                "Fuel Used (L)": round(random.uniform(2, 7), 1),
                "Load Cycles": random.randint(1, 15),
                "Idling Time (min)": random.randint(5, 60),
                "Engine Hours": round(random.uniform(1523, 1535), 1),
            }
        ]
    )

    cnn_numeric_input = cnn_input.values.astype(float)

    predictions = anomaly_cnn_model.predict(cnn_numeric_input)[0]
    anomaly_flags = (predictions > 0.5).astype(int).tolist()

    return {
        "CNN Features": cnn_input.iloc[0].to_dict(),
        "Anomaly Prediction Flags": anomaly_flags,
    }


def generate_combined_payload():
    task_input = generate_task_time_input()
    predicted_time = fetch_task_time_prediction(task_input)

    combined = {
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Machine ID": "EXC001",
        **task_input,
        "Predicted Task Completion Time (min)": predicted_time,
    }

    combined.update(generate_anomaly_prediction_payload())
    return combined


def on_connect(client, userdata, flags, rc):
    print("Simulator connected to broker.")
    client.subscribe(ALERT_TOPIC)


def on_message(client, userdata, msg):
    print("🔔 Alert received:", msg.payload.decode())


def run_simulator():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, 1883, 60)
    client.loop_start()

    while True:
        combined_data = generate_combined_payload()
        if combined_data["Predicted Task Completion Time (min)"] is not None:
            client.publish(DATA_TOPIC, json.dumps(combined_data))
            print("📤 Sent:", combined_data)
        else:
            print("⚠️ Skipped sending due to API prediction failure.")
        time.sleep(2)


if __name__ == "__main__":
    run_simulator()
