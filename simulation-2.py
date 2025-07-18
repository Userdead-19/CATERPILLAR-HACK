#!/usr/bin/env python3
"""
Task Time Prediction API Client - FIXED VERSION
Tests the task time prediction endpoint of the Flask app
"""

import requests
import json
import random
import time
import numpy as np
from datetime import datetime

# Configuration
FLASK_API_URL = "http://localhost:5000"
PREDICT_ENDPOINT = f"{FLASK_API_URL}/predict-task-time"


class TaskTimePredictionClient:
    def __init__(self):
        # Define realistic task scenarios
        self.task_scenarios = {
            "Excavation": {
                "Flat": {"complexity": "Medium", "urgency": "Normal"},
                "Hilly": {"complexity": "High", "urgency": "High"},
                "Rocky": {"complexity": "Very High", "urgency": "High"},
                "Muddy": {"complexity": "High", "urgency": "Normal"},
            },
            "Loading": {
                "Flat": {"complexity": "Low", "urgency": "Normal"},
                "Hilly": {"complexity": "Medium", "urgency": "Normal"},
                "Rocky": {"complexity": "High", "urgency": "High"},
                "Muddy": {"complexity": "Medium", "urgency": "Normal"},
            },
            "Dozing": {
                "Flat": {"complexity": "Low", "urgency": "Normal"},
                "Hilly": {"complexity": "Medium", "urgency": "Normal"},
                "Rocky": {"complexity": "High", "urgency": "High"},
                "Muddy": {"complexity": "Medium", "urgency": "Normal"},
            },
            "Grading": {
                "Flat": {"complexity": "Low", "urgency": "Normal"},
                "Hilly": {"complexity": "Medium", "urgency": "Normal"},
                "Rocky": {"complexity": "High", "urgency": "High"},
                "Muddy": {"complexity": "Medium", "urgency": "Normal"},
            },
            "Transport": {  # Added Transport task type
                "Flat": {"complexity": "Low", "urgency": "Normal"},
                "Hilly": {"complexity": "Medium", "urgency": "Normal"},
                "Rocky": {"complexity": "High", "urgency": "High"},
                "Muddy": {"complexity": "Medium", "urgency": "Normal"},
            },
        }

        self.operators = [f"OP_{i:03d}" for i in range(1, 11)]
        self.machines = [f"MACHINE_{i:03d}" for i in range(1, 6)]

    def generate_task_request(self):
        """Generate a realistic task prediction request with correct feature names"""

        # Select random task type and terrain
        task_type = random.choice(list(self.task_scenarios.keys()))
        terrain = random.choice(list(self.task_scenarios[task_type].keys()))

        # Generate realistic feature values matching your exact dataset columns
        task_data = {
            # Numerical features
            "Engine_Hours": random.randint(100, 5000),  # Total engine hours
            "Fuel_Used_L": random.randint(50, 500),  # Fuel used in liters
            "Load_Cycles": random.randint(5, 50),  # Number of load cycles
            "Idling_Time_min": random.randint(10, 120),  # Idling time in minutes
            "Seatbelt_Status": random.choice([0, 1]),  # Binary: 0=off, 1=on
            "Safety_Alert": random.choice([0, 1]),  # Binary: 0=no alert, 1=alert
            "Operator_ID": random.randint(1, 50),  # Operator ID as integer
            "Weather_Temp_C": random.randint(15, 35),  # Temperature in Celsius
            "Weather_Rainfall_mm": random.randint(0, 50),  # Rainfall in mm
            "Weather_Wind_kmph": random.randint(0, 25),  # Wind speed in km/h
            # Categorical features (exact column names from your dataset)
            "Terrain": terrain,
            "Task_Type": task_type,
            # Note: Task_Completion_Time_min is the target variable, not included in prediction request
        }

        return task_data

    def predict_task_time(self, task_data):
        """Send prediction request to Flask API"""
        try:
            headers = {"Content-Type": "application/json"}
            response = requests.post(
                PREDICT_ENDPOINT, json=task_data, headers=headers, timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("estimated_time")
            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"❌ Request Error: {e}")
            return None

    def test_single_prediction(self):
        """Test a single task time prediction"""
        print("🧪 Testing Single Task Time Prediction")
        print("=" * 50)

        task_data = self.generate_task_request()

        print("📋 Task Details:")
        print(f"   Task Type: {task_data['Task_Type']}")
        print(f"   Terrain: {task_data['Terrain']}")
        print(f"   Engine Hours: {task_data['Engine_Hours']}")
        print(f"   Fuel Used: {task_data['Fuel_Used_L']} L")
        print(f"   Load Cycles: {task_data['Load_Cycles']}")
        print(f"   Idling Time: {task_data['Idling_Time_min']} min")
        print(f"   Operator ID: {task_data['Operator_ID']}")
        print(f"   Seatbelt Status: {task_data['Seatbelt_Status']}")
        print(f"   Safety Alert: {task_data['Safety_Alert']}")
        print(f"   Temperature: {task_data['Weather_Temp_C']}°C")
        print(f"   Rainfall: {task_data['Weather_Rainfall_mm']} mm")
        print(f"   Wind Speed: {task_data['Weather_Wind_kmph']} km/h")

        print("\n⏳ Requesting prediction...")

        estimated_time = self.predict_task_time(task_data)

        if estimated_time is not None:
            print(f"✅ Estimated Task Time: {estimated_time:.2f} minutes")
            print(f"   Estimated Duration: {estimated_time/60:.1f} hours")
        else:
            print("❌ Prediction failed")

        return estimated_time

    def test_batch_predictions(self, num_tests=10):
        """Test multiple task time predictions"""
        print(f"🧪 Testing Batch Task Time Predictions ({num_tests} samples)")
        print("=" * 60)

        results = []

        for i in range(num_tests):
            task_data = self.generate_task_request()
            estimated_time = self.predict_task_time(task_data)

            if estimated_time is not None:
                results.append(
                    {
                        "task_type": task_data["Task_Type"],
                        "terrain": task_data["Terrain"],
                        "estimated_time": estimated_time,
                    }
                )

                print(
                    f"✅ Test {i+1:2d}: {task_data['Task_Type']:12} | "
                    f"{task_data['Terrain']:6} | "
                    f"{estimated_time:6.1f} min"
                )
            else:
                print(f"❌ Test {i+1:2d}: Prediction failed")

            time.sleep(0.5)  # Small delay between requests

        # Summary statistics
        if results:
            times = [r["estimated_time"] for r in results]
            print("\n📊 Batch Results Summary:")
            print("=" * 40)
            print(f"Successful Predictions: {len(results)}/{num_tests}")
            print(f"Average Time: {np.mean(times):.2f} minutes")
            print(f"Min Time: {np.min(times):.2f} minutes")
            print(f"Max Time: {np.max(times):.2f} minutes")
            print(f"Std Deviation: {np.std(times):.2f} minutes")

            # Group by task type
            task_types = {}
            for result in results:
                task_type = result["task_type"]
                if task_type not in task_types:
                    task_types[task_type] = []
                task_types[task_type].append(result["estimated_time"])

            print(f"\n📈 Average Time by Task Type:")
            for task_type, times in task_types.items():
                print(f"   {task_type:12}: {np.mean(times):6.1f} min")

        return results

    def test_edge_cases(self):
        """Test edge cases and unusual scenarios"""
        print("🧪 Testing Edge Cases")
        print("=" * 40)

        edge_cases = [
            {
                "name": "High Load Rocky Terrain",
                "data": {
                    "Engine_Hours": 4500,
                    "Fuel_Used_L": 450,
                    "Load_Cycles": 45,
                    "Idling_Time_min": 100,
                    "Seatbelt_Status": 1,
                    "Safety_Alert": 1,
                    "Operator_ID": 25,
                    "Weather_Temp_C": 35,
                    "Weather_Rainfall_mm": 40,
                    "Weather_Wind_kmph": 25,
                    "Terrain": "Rocky",
                    "Task_Type": "Excavation",
                },
            },
            {
                "name": "Light Load Flat Terrain",
                "data": {
                    "Engine_Hours": 200,
                    "Fuel_Used_L": 80,
                    "Load_Cycles": 8,
                    "Idling_Time_min": 15,
                    "Seatbelt_Status": 1,
                    "Safety_Alert": 0,
                    "Operator_ID": 5,
                    "Weather_Temp_C": 22,
                    "Weather_Rainfall_mm": 0,
                    "Weather_Wind_kmph": 5,
                    "Terrain": "Flat",
                    "Task_Type": "Loading",
                },
            },
            {
                "name": "Transport Mixed Conditions",
                "data": {
                    "Engine_Hours": 2500,
                    "Fuel_Used_L": 200,
                    "Load_Cycles": 20,
                    "Idling_Time_min": 60,
                    "Seatbelt_Status": 1,
                    "Safety_Alert": 0,
                    "Operator_ID": 15,
                    "Weather_Temp_C": 28,
                    "Weather_Rainfall_mm": 15,
                    "Weather_Wind_kmph": 15,
                    "Terrain": "Hilly",
                    "Task_Type": "Transport",
                },
            },
        ]

        for case in edge_cases:
            print(f"\n📋 {case['name']}:")
            estimated_time = self.predict_task_time(case["data"])

            if estimated_time is not None:
                print(f"   ✅ Estimated Time: {estimated_time:.2f} minutes")
            else:
                print(f"   ❌ Prediction failed")

    def interactive_prediction(self):
        """Interactive prediction mode"""
        print("🎯 Interactive Task Time Prediction")
        print("=" * 40)

        # Get user input
        task_type = input(
            "Task Type (Excavation/Loading/Dozing/Grading/Transport): "
        ).strip()
        terrain = input("Terrain (Flat/Hilly/Rocky/Muddy): ").strip()

        if not task_type or not terrain:
            print("❌ Invalid input, using random values")
            task_data = self.generate_task_request()
        else:
            task_data = {
                "Engine_Hours": random.randint(100, 5000),
                "Fuel_Used_L": random.randint(50, 500),
                "Load_Cycles": random.randint(5, 50),
                "Idling_Time_min": random.randint(10, 120),
                "Seatbelt_Status": random.choice([0, 1]),
                "Safety_Alert": random.choice([0, 1]),
                "Operator_ID": random.randint(1, 50),
                "Weather_Temp_C": random.randint(15, 35),
                "Weather_Rainfall_mm": random.randint(0, 50),
                "Weather_Wind_kmph": random.randint(0, 25),
                "Terrain": terrain,
                "Task_Type": task_type,
            }

        print(
            f"\n📋 Generated Task: {task_data['Task_Type']} on {task_data['Terrain']} terrain"
        )
        estimated_time = self.predict_task_time(task_data)

        if estimated_time is not None:
            print(
                f"✅ Estimated Time: {estimated_time:.2f} minutes ({estimated_time/60:.1f} hours)"
            )
        else:
            print("❌ Prediction failed")

    def run_client(self):
        """Main client interface"""
        print("🚀 Task Time Prediction API Client - FIXED VERSION")
        print("Make sure your Flask app is running on http://localhost:5000")
        print("=" * 60)

        while True:
            print("\n🎯 Task Time Prediction Menu")
            print("=" * 40)
            print("1. Single prediction test")
            print("2. Batch predictions (10 samples)")
            print("3. Edge cases test")
            print("4. Interactive prediction")
            print("5. Exit")

            choice = input("\nEnter your choice (1-5): ").strip()

            if choice == "1":
                self.test_single_prediction()
            elif choice == "2":
                self.test_batch_predictions(10)
            elif choice == "3":
                self.test_edge_cases()
            elif choice == "4":
                self.interactive_prediction()
            elif choice == "5":
                print("👋 Exiting client...")
                break
            else:
                print("❌ Invalid choice, please try again")

            time.sleep(1)


if __name__ == "__main__":
    client = TaskTimePredictionClient()
    client.run_client()
