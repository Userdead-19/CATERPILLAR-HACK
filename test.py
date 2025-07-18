import pandas as pd
import random

# Load existing dataset
df = pd.read_csv("dataset.csv")

# Define required extra fields and default/random value generators
extra_fields = {
    "Tire Pressure (PSI)": lambda: round(random.uniform(30, 80), 1),
    "Hydraulic Pressure (Bar)": lambda: round(random.uniform(150, 250), 1),
    "Oil Temperature (°C)": lambda: round(random.uniform(60, 120), 1),
    "Battery Voltage (V)": lambda: round(random.uniform(11.5, 14.8), 2),
}

# Add missing columns if not already present
for field, value_fn in extra_fields.items():
    if field not in df.columns:
        print(f"Adding missing field: {field}")
        df[field] = [value_fn() for _ in range(len(df))]

# Save updated dataset
df.to_csv("dataset_with_extra_fields.csv", index=False)

print("✅ Updated dataset saved as dataset_with_extra_fields.csv")
