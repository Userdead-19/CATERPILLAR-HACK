import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("dataset.csv")  # Ensure this path is correct

# Define feature and target columns
feature_columns = ["Fuel Used (L)", "Load Cycles", "Idling Time (min)", "Engine Hours"]

target_columns = [
    "Excessive Fuel Consumption",
    "High Idling Time",
    "Seatbelt Violation",
    "Engine Hour Anomaly",
    "Low Load Cycles",
    "High Load Cycles",
    "Multiple Abnormalities",
]

# Prepare feature matrix X and label matrix y
X = df[feature_columns].values
y = df[target_columns].fillna(0).astype(int).values  # Ensuring binary format

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build neural network model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(len(feature_columns),)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(len(target_columns), activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Save model file
model.save("multi_anomaly_nn_model.h5")

# Save associated metadata as a package (features + labels)
model_metadata = {
    "features": feature_columns,
    "labels": target_columns,
}
joblib.dump(model_metadata, "multi_anomaly_nn_meta.pkl")

print("✅ Neural network model and metadata saved successfully.")
