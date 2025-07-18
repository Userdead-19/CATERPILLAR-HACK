# trainmodel.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import numpy as np


def train_and_save_model(data_path):
    # Load dataset
    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop("Task_Completion_Time_min", axis=1)
    y = df["Task_Completion_Time_min"]

    # Categorical and numerical features
    cat_features = ["Terrain", "Task_Type"]

    # Custom preprocessing pipeline WITHOUT prefixes
    # Added handle_unknown='ignore' to handle unknown categories
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features)
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,  # This removes the prefixes!
    )

    # Full pipeline with model
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", XGBRegressor(random_state=42))]
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit pipeline
    pipeline.fit(X_train, y_train)

    # Predict and evaluate
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print("Mean Absolute Error:", mae)

    # Save the pipeline as a single .pkl file
    joblib.dump(pipeline, "task_time_prediction.pkl")
    print("Pipeline with model saved as task_time_prediction.pkl")

    # Optionally save feature names separately if needed elsewhere
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    joblib.dump(feature_names, "task_time_features.pkl")
    print("Feature names saved as task_time_features.pkl")

    print("Feature names after preprocessing:", feature_names)

    # Check what terrain and task types are available in the training data
    print("\nAvailable categories in training data:")
    print("Terrain types:", sorted(df["Terrain"].unique()))
    print("Task types:", sorted(df["Task_Type"].unique()))

    # Test the pipeline with sample data
    print("\n" + "=" * 50)
    print("Testing the trained pipeline:")

    # Create a sample test case with KNOWN categories
    sample_data = pd.DataFrame(
        {
            "Engine_Hours": [2.5],  # Using values similar to training data
            "Fuel_Used_L": [50.0],
            "Load_Cycles": [100],
            "Idling_Time_min": [45],
            "Seatbelt_Status": [1],
            "Safety_Alert": [0],
            "Operator_ID": [15],
            "Weather_Temp_C": [25],
            "Weather_Rainfall_mm": [10],
            "Weather_Wind_kmph": [15],
            "Terrain": ["Rocky"],  # Using a known terrain type
            "Task_Type": ["Loading"],  # Using a known task type
        }
    )

    # Test prediction
    test_prediction = pipeline.predict(sample_data)
    print(f"Sample prediction: {test_prediction[0]:.2f} minutes")

    # Show what the preprocessor outputs
    processed_sample = pipeline.named_steps["preprocessor"].transform(sample_data)
    print(f"Processed sample shape: {processed_sample.shape}")
    print(f"Feature names: {list(feature_names)}")

    # Test with unknown category to show it handles it gracefully
    print("\n" + "=" * 30)
    print("Testing with unknown category:")

    unknown_sample = pd.DataFrame(
        {
            "Engine_Hours": [2.5],
            "Fuel_Used_L": [50.0],
            "Load_Cycles": [100],
            "Idling_Time_min": [45],
            "Seatbelt_Status": [1],
            "Safety_Alert": [0],
            "Operator_ID": [15],
            "Weather_Temp_C": [25],
            "Weather_Rainfall_mm": [10],
            "Weather_Wind_kmph": [15],
            "Terrain": ["Hilly"],  # Unknown terrain type
            "Task_Type": ["Excavation"],  # Unknown task type
        }
    )

    unknown_prediction = pipeline.predict(unknown_sample)
    print(f"Prediction with unknown categories: {unknown_prediction[0]:.2f} minutes")
    print("Note: Unknown categories are treated as all zeros in one-hot encoding")

    return pipeline


if __name__ == "__main__":
    pipeline = train_and_save_model(
        "task_time_dataset.csv"
    )  # Replace with your CSV filename
