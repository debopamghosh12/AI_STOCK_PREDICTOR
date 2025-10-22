import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error # <-- Import MAE
import joblib
import os
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import load_model # <-- To load existing model if needed

# --- Constants ---
WINDOW_SIZE = 60
HORIZON_SIZE = 7
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
TARGET_COLUMN_INDEX = 3
STOCK_TO_EVALUATE = 'AAPL' # <-- Choose the stock you want to test
TRAIN_TEST_SPLIT_RATIO = 0.8 # Use 80% for training, 20% for testing

model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# --- Functions (create_sequences and build_lstm_model are the same as before) ---
def create_sequences(data, window_size, horizon_size, target_index):
    X, y = [], []
    for i in range(len(data) - window_size - horizon_size + 1):
        window = data[i:(i + window_size)]
        horizon = data[(i + window_size):(i + window_size + horizon_size), target_index]
        X.append(window)
        y.append(horizon)
    return np.array(X), np.array(y)

def build_lstm_model(window_size, n_features, horizon_size):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(window_size, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=horizon_size))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- Main Evaluation Script ---
print(f"--- Evaluating LSTM model performance for {STOCK_TO_EVALUATE} ---")
start_time = time.time()

# 1. Fetch Data
try:
    data = yf.download(STOCK_TO_EVALUATE, period='7y', progress=False)
    data = data[FEATURES]
    if data.empty or data.isnull().values.any() or len(data) < 200:
        print(f"Not enough valid data for {STOCK_TO_EVALUATE}. Exiting.")
        exit()
except Exception as e:
    print(f"Error fetching data: {e}. Exiting.")
    exit()

# 2. Split Data into Training and Test Sets
split_index = int(len(data) * TRAIN_TEST_SPLIT_RATIO)
train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]
print(f"Training data points: {len(train_data)}")
print(f"Testing data points: {len(test_data)}")

# 3. Scale Data
# IMPORTANT: Fit scaler ONLY on training data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data)
# Use the SAME scaler (fitted on train) to transform test data
scaled_test_data = scaler.transform(test_data)

# 4. Create Sequences for Training and Testing
X_train, y_train = create_sequences(scaled_train_data, WINDOW_SIZE, HORIZON_SIZE, TARGET_COLUMN_INDEX)
X_test, y_test_actual_scaled = create_sequences(scaled_test_data, WINDOW_SIZE, HORIZON_SIZE, TARGET_COLUMN_INDEX)

if X_train.shape[0] == 0 or X_test.shape[0] == 0:
    print("Could not create sequences. Not enough data after split.")
    exit()

n_features = X_train.shape[2]

# 5. Build and Train Model (or Load Existing)
model_filename = f"{model_dir}/model_{STOCK_TO_EVALUATE}.keras"

# Option A: Train a new model just for evaluation
print("Training new model for evaluation...")
model = build_lstm_model(WINDOW_SIZE, n_features, HORIZON_SIZE)
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1) # verbose=1 shows training progress

# Option B: Load your previously trained model (if you want to evaluate that exact one)
# print(f"Loading existing model: {model_filename}...")
# if os.path.exists(model_filename):
#     model = load_model(model_filename)
# else:
#     print("Model file not found. Train a new one or check path.")
#     exit()

# 6. Make Predictions on the Test Set
print("Making predictions on test data...")
y_pred_scaled = model.predict(X_test)

# 7. Inverse Transform Predictions and Actual Values
# We need to use the 'trick' again to unscale the target column

# Unscale Predictions
dummy_pred = np.zeros((len(y_pred_scaled), n_features))
dummy_pred[:, TARGET_COLUMN_INDEX] = y_pred_scaled[:, 0] # Use only the first day for MAE calculation for simplicity, or loop through all 7
y_pred_unscaled = scaler.inverse_transform(dummy_pred)[:, TARGET_COLUMN_INDEX]

# Unscale Actual Test Values
dummy_actual = np.zeros((len(y_test_actual_scaled), n_features))
dummy_actual[:, TARGET_COLUMN_INDEX] = y_test_actual_scaled[:, 0] # Use only the first day
y_test_actual_unscaled = scaler.inverse_transform(dummy_actual)[:, TARGET_COLUMN_INDEX]

# --- Calculate MAE for the 1st prediction day ---
mae = mean_absolute_error(y_test_actual_unscaled, y_pred_unscaled)

end_time = time.time()
print("\n--- Evaluation Complete ---")
print(f"Stock Evaluated: {STOCK_TO_EVALUATE}")
print(f"Time Taken: {end_time - start_time:.2f} seconds")
print(f"**Mean Absolute Error (MAE) for Day 1 Forecast: ${mae:.2f}**")
print("(This means, on average, the Day 1 prediction was off by this amount in dollars on the test data)")