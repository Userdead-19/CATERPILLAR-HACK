# trainmodel.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import joblib


def train_and_save_model(data_path):
    # Load dataset
    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop("Task_Completion_Time_min", axis=1)
    y = df["Task_Completion_Time_min"]

    # Categorical and numerical features
    cat_features = ["Terrain", "Task_Type"]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(drop="first"), cat_features)],
        remainder="passthrough",
    )

    # Full pipeline with model
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", XGBRegressor())]
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


if __name__ == "__main__":
    train_and_save_model("task_time_dataset.csv")  # Replace with your CSV filename
