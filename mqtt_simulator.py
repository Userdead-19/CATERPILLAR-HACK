#!/usr/bin/env python3
"""
IoT Device Simulator - MQTT Data Sender
Simulates IoT machines sending data to the Flask app via MQTT
"""

import paho.mqtt.client as mqtt
import json
import random
import time
import numpy as np
from datetime import datetime, timedelta
import requests
import threading

# Configuration
MQTT_BROKER = "test.mosquitto.org"
MQTT_TOPIC = "iot/machine/data"
NOTIFY_TOPIC = "iot/machine/alerts"
FLASK_API_URL = "http://localhost:5000"


class IoTDeviceSimulator:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.running = False
        self.machine_ids = [f"MACHINE_{i:03d}" for i in range(1, 6)]
        self.operators = [f"OP_{i:03d}" for i in range(1, 11)]
        self.locations = ["Site_A", "Site_B", "Site_C", "Site_D"]

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("✅ Connected to MQTT broker")
            # Subscribe to alerts to monitor anomaly notifications
            client.subscribe(NOTIFY_TOPIC)
            print(f"📡 Subscribed to alerts topic: {NOTIFY_TOPIC}")
        else:
            print(f"❌ Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        """Handle incoming alert messages"""
        try:
            if msg.topic == NOTIFY_TOPIC:
                alert = json.loads(msg.payload.decode())
                print(f"🚨 ALERT RECEIVED: {alert}")
                self.display_alert(alert)
        except Exception as e:
            print(f"❌ Error processing alert: {e}")

    def display_alert(self, alert):
        """Display alert in a formatted way"""
        print("=" * 60)
        print("🚨 ANOMALY ALERT DETECTED")
        print("=" * 60)
        print(f"Machine ID: {alert.get('machine_id', 'Unknown')}")
        print(f"Timestamp: {alert.get('timestamp', 'Unknown')}")
        print(f"Fuel Used: {alert.get('Fuel Used (L)', 'N/A'):.2f}L")
        print(f"Load Cycles: {alert.get('Load Cycles', 'N/A'):.0f}")
        print(f"Idling Time: {alert.get('Idling Time (min)', 'N/A'):.1f} min")
        print(f"Engine Hours: {alert.get('Engine Hours', 'N/A'):.1f} hrs")

        # Show detected anomaly types
        anomaly_types = []
        for key, value in alert.items():
            if key.endswith(
                (
                    "Consumption",
                    "Time",
                    "Violation",
                    "Anomaly",
                    "Cycles",
                    "Abnormalities",
                )
            ):
                if value:
                    anomaly_types.append(key)

        if anomaly_types:
            print(f"Anomaly Types: {', '.join(anomaly_types)}")
        print("=" * 60)

    def generate_machine_data(self, include_anomalies=True):
        """Generate realistic machine data"""

        # Base normal data
        data = {
            "machine_id": random.choice(self.machine_ids),
            "timestamp": datetime.now().isoformat(),
            "operator_id": random.choice(self.operators),
            "location": random.choice(self.locations),
            "Fuel Used (L)": np.random.normal(25, 5),  # Normal: 15-35L
            "Load Cycles": np.random.normal(150, 30),  # Normal: 100-200 cycles
            "Idling Time (min)": np.random.normal(45, 10),  # Normal: 30-60 min
            "Engine Hours": np.random.normal(8, 2),  # Normal: 6-10 hours
            "temperature": np.random.normal(22, 5),
            "humidity": np.random.normal(55, 10),
            "vibration_level": np.random.normal(3, 1),
            "oil_pressure": np.random.normal(35, 5),
        }

        # Introduce anomalies for some samples (30% chance)
        if include_anomalies and random.random() < 0.3:
            anomaly_type = random.choice(
                [
                    "high_fuel",
                    "excessive_idling",
                    "engine_hours",
                    "low_load",
                    "high_load",
                    "multiple",
                ]
            )

            print(f"🔥 Injecting {anomaly_type} anomaly...")

            if anomaly_type == "high_fuel":
                data["Fuel Used (L)"] = np.random.normal(50, 8)  # Excessive fuel
            elif anomaly_type == "excessive_idling":
                data["Idling Time (min)"] = np.random.normal(90, 15)  # High idling
            elif anomaly_type == "engine_hours":
                data["Engine Hours"] = np.random.normal(15, 3)  # Excessive hours
            elif anomaly_type == "low_load":
                data["Load Cycles"] = np.random.normal(50, 10)  # Low load
            elif anomaly_type == "high_load":
                data["Load Cycles"] = np.random.normal(300, 20)  # High load
            elif anomaly_type == "multiple":
                data["Fuel Used (L)"] = np.random.normal(45, 5)
                data["Idling Time (min)"] = np.random.normal(85, 10)
                data["Load Cycles"] = np.random.normal(280, 15)

        # Ensure positive values
        data["Fuel Used (L)"] = max(0, data["Fuel Used (L)"])
        data["Load Cycles"] = max(0, data["Load Cycles"])
        data["Idling Time (min)"] = max(0, data["Idling Time (min)"])
        data["Engine Hours"] = max(0, data["Engine Hours"])

        return data

    def send_data_batch(self, num_samples=10):
        """Send a batch of data samples"""
        print(f"📤 Sending batch of {num_samples} samples...")

        for i in range(num_samples):
            data = self.generate_machine_data()

            # Send via MQTT
            try:
                self.client.publish(MQTT_TOPIC, json.dumps(data))
                print(
                    f"✅ Sent sample {i+1:2d}: {data['machine_id']} | "
                    f"Fuel: {data['Fuel Used (L)']:.1f}L | "
                    f"Cycles: {data['Load Cycles']:.0f} | "
                    f"Idling: {data['Idling Time (min)']:.1f}min"
                )
            except Exception as e:
                print(f"❌ Failed to send sample {i+1}: {e}")

            time.sleep(0.5)  # Small delay between samples

    def start_real_time_simulation(self, duration=120, interval=3):
        """Start real-time data simulation"""
        print(f"🚀 Starting real-time simulation for {duration} seconds...")
        print(f"📊 Sending data every {interval} seconds")
        print("=" * 60)

        self.running = True
        start_time = time.time()
        sample_count = 0

        while self.running and (time.time() - start_time) < duration:
            data = self.generate_machine_data()
            sample_count += 1

            try:
                self.client.publish(MQTT_TOPIC, json.dumps(data))
                print(
                    f"📡 [{sample_count:03d}] Sent: {data['machine_id']} | "
                    f"F:{data['Fuel Used (L)']:.1f}L | "
                    f"C:{data['Load Cycles']:.0f} | "
                    f"I:{data['Idling Time (min)']:.1f}m | "
                    f"E:{data['Engine Hours']:.1f}h"
                )
            except Exception as e:
                print(f"❌ Error sending sample {sample_count}: {e}")

            time.sleep(interval)

        print(f"\n🏁 Simulation complete! Sent {sample_count} samples")

    def test_flask_api(self):
        """Test Flask API endpoints"""
        print("🧪 Testing Flask API endpoints...")
        print("=" * 60)

        # Test /latest endpoint
        try:
            response = requests.get(f"{FLASK_API_URL}/latest")
            if response.status_code == 200:
                latest_data = response.json()
                print("✅ /latest endpoint working")
                print(f"Latest data: {latest_data}")
            else:
                print(f"❌ /latest endpoint failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Error testing /latest: {e}")

        # Test /all endpoint
        try:
            response = requests.get(f"{FLASK_API_URL}/all")
            if response.status_code == 200:
                all_data = response.json()
                print(f"✅ /all endpoint working - {len(all_data)} records")
            else:
                print(f"❌ /all endpoint failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Error testing /all: {e}")

        print("=" * 60)

    def connect_and_run(self):
        """Connect to MQTT broker and start simulation"""
        try:
            self.client.connect(MQTT_BROKER, 1883, 60)
            self.client.loop_start()

            # Wait for connection
            time.sleep(2)

            # Menu system
            while True:
                print("\n🤖 IoT Device Simulator Menu")
                print("=" * 40)
                print("1. Send batch data (10 samples)")
                print("2. Start real-time simulation (120s)")
                print("3. Test Flask API endpoints")
                print("4. Send single sample")
                print("5. Exit")

                choice = input("\nEnter your choice (1-5): ").strip()

                if choice == "1":
                    self.send_data_batch(10)
                elif choice == "2":
                    self.start_real_time_simulation(120, 3)
                elif choice == "3":
                    self.test_flask_api()
                elif choice == "4":
                    data = self.generate_machine_data()
                    self.client.publish(MQTT_TOPIC, json.dumps(data))
                    print(f"📤 Sent single sample: {data['machine_id']}")
                elif choice == "5":
                    print("👋 Exiting simulator...")
                    self.running = False
                    break
                else:
                    print("❌ Invalid choice, please try again")

                time.sleep(1)

        except Exception as e:
            print(f"❌ Connection error: {e}")
        finally:
            self.client.loop_stop()
            self.client.disconnect()


if __name__ == "__main__":
    print("🚀 IoT Device Simulator Starting...")
    print("Make sure your Flask app is running on http://localhost:5000")
    print("=" * 60)

    simulator = IoTDeviceSimulator()
    simulator.connect_and_run()
