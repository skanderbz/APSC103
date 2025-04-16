import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from models.lstm_model import build_lstm_model

# Load the dataset
df = pd.read_excel('data/plant_data.xlsx')

# Select relevant features and target
features = ['NH4', 'NOx', 'DO', 'Temp', 'pH']
target = ['N2O_emission']

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[features + target])

# Create sequences for LSTM (e.g. past 30 days)
sequence_length = 30
X, y = [], []

for i in range(sequence_length, len(data_scaled)):
    X.append(data_scaled[i-sequence_length:i, :-1])  # past 30 days of features
    y.append(data_scaled[i, -1])                     # N2O on current day

X = np.array(X)
y = np.array(y)

# Train-test split (no shuffle due to time series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build and train model
model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
history = model.fit(X_train, y_train, epochs=80, batch_size=16, validation_split=0.2)

# Predict
y_pred = model.predict(X_test)
model.save_weights("model.weights.h5")

# Plot predictions
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='True')
plt.plot(y_pred, label='Predicted')
plt.title('N₂O Emission Prediction')
plt.xlabel('Days')
plt.ylabel('Normalized Emissions')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()