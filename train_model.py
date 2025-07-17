import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Load dataset
df = pd.read_csv("dataset.csv")

# Select features for training
features = df[["Fuel Used (L)", "Load Cycles", "Idling Time (min)", "Engine Hours"]]

# Train model
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(features)

# Save the model
joblib.dump(model, "anomaly_model.pkl")
print("Model saved as anomaly_model.pkl")
