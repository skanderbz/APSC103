import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from models.lstm_model import build_lstm_model
import os

# --- CONFIG ---
UI_PATH = "data/user_inputs.xlsx"
PLANT_DATA_PATH = "data/plant_data.xlsx"
MODEL_WEIGHTS_PATH = "model.weights.h5"
FEATURES = ['NH4', 'NOx', 'DO', 'Temp', 'pH']
TARGET = ['N2O_emission']

# --- LOAD TRAINING DATA FOR SCALER FITTING ---
try:
    df = pd.read_excel(PLANT_DATA_PATH)
except FileNotFoundError:
    print(f"❌ Could not find {PLANT_DATA_PATH}")
    exit()

scaler = MinMaxScaler()
scaler.fit(df[FEATURES + TARGET])

# --- LOAD LSTM MODEL ---
model = build_lstm_model(input_shape=(30, len(FEATURES)))
if os.path.exists(MODEL_WEIGHTS_PATH):
    model.load_weights(MODEL_WEIGHTS_PATH)
else:
    print(f"❌ Model weights not found: {MODEL_WEIGHTS_PATH}")
    exit()

# --- LOAD USER INPUTS ---
try:
    user_df = pd.read_excel(UI_PATH)
except FileNotFoundError:
    print(f"❌ Could not find {UI_PATH}")
    exit()

# Preserve original Date column (optional)
dates = user_df["Date"] if "Date" in user_df.columns else pd.NaT

# --- PREPARE INPUT DATA ---
input_df = user_df[FEATURES].copy()

# Fill missing values with mean from plant data
for col in FEATURES:
    if col not in input_df or input_df[col].isnull().all():
        input_df[col] = df[col].mean()
    else:
        input_df[col] = input_df[col].fillna(df[col].mean())

# Append dummy N2O column for scaler compatibility
temp_df = input_df.copy()
temp_df["N2O_emission"] = 0
scaled_inputs = scaler.transform(temp_df)[:, :-1]  # remove dummy target post-scaling

# Repeat each row 30x to simulate time series context
sequences = np.array([
    np.repeat(row[np.newaxis, :], 30, axis=0)
    for row in scaled_inputs
])

# --- PREDICT ---
predictions = model.predict(sequences).flatten()

# --- UPDATE EXCEL FILE ---
user_df["Predicted N2O"] = predictions
if "Date" in user_df.columns:
    user_df["Date"] = dates  # keep the Date column unchanged

user_df.to_excel(UI_PATH, index=False)
print("✅ Predictions added to Excel successfully.")
