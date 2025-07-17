import json
import random
import time
import paho.mqtt.client as mqtt

MQTT_BROKER = "test.mosquitto.org"
DATA_TOPIC = "iot/machine/data"
ALERT_TOPIC = "iot/machine/alerts"


def generate_payload():
    return {
        "Machine ID": "EXC001",
        "Operator ID": "OP1001",
        "Engine Hours": round(random.uniform(1523, 1535), 1),
        "Fuel Used (L)": round(random.uniform(2, 7), 1),
        "Load Cycles": random.randint(1, 15),
        "Idling Time (min)": random.randint(5, 60),
        "Seatbelt Status": random.choice(["Fastened", "Unfastened"]),
        "Safety Alert Triggered": random.choice(["Yes", "No"]),
    }


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
        data = generate_payload()
        client.publish(DATA_TOPIC, json.dumps(data))
        print("📤 Sent:", data)
        time.sleep(2)


if __name__ == "__main__":
    run_simulator()
