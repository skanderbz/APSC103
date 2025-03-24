from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(1)  # Output: predicted N2O emission
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

