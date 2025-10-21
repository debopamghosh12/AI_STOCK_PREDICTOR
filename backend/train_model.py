import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


WINDOW_SIZE = 60
HORIZON_SIZE = 7

FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']

TARGET_COLUMN_INDEX = 3 


STOCKS_TO_TRAIN = [
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 'TSLA', 'META', 'AMD', 'INTC', 'CRM',
    'ORCL', 'IBM', 'QCOM', 'CSCO', 'JPM', 'V', 'BAC', 'MA', 'WFC', 'GS',
    'JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'WMT', 'COST', 'HD', 'NKE', 'MCD',
    'NFLX', 'DIS', 'CMCSA', 'XOM', 'CVX', 'BA', 'CAT', 'GE', 'KO', 'PEP',
    'PG', 'T', 'VZ', 'UPS', 'FDX', 'SBUX', 'TGT', 'LMT', 'PYPL'
]

model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

def create_sequences(data, window_size, horizon_size, target_index):
    """
    Creates sequences from multivariate data.
    X = (samples, window_size, n_features)
    y = (samples, horizon_size)
    """
    X, y = [], []
    for i in range(len(data) - window_size - horizon_size + 1):
        window = data[i:(i + window_size)]
        
        horizon = data[(i + window_size):(i + window_size + horizon_size), target_index]
        X.append(window)
        y.append(horizon)
    return np.array(X), np.array(y)

def build_lstm_model(window_size, n_features, horizon_size):
    """Builds a Multivariate LSTM model."""
    model = Sequential()
    
    
    model.add(LSTM(units=50, return_sequences=True, input_shape=(window_size, n_features)))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=25))
    
    
    model.add(Dense(units=horizon_size))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


print(f"\n--- Starting MULTIVARIATE LSTM model training for {len(STOCKS_TO_TRAIN)} stocks ---")
print(f"This will take a VERY long time...")

start_total_time = time.time()
successful_models = 0
failed_tickers = []

if 'KO' not in STOCKS_TO_TRAIN: STOCKS_TO_TRAIN.append('KO')

for i, ticker in enumerate(STOCKS_TO_TRAIN):
    print(f"\nTraining model {i+1}/{len(STOCKS_TO_TRAIN)}: {ticker} ...", end="", flush=True)
    stock_start_time = time.time()
    
    
    try:
        data = yf.download(ticker, period='7y', progress=False)
        data = data[FEATURES] # Keep only our 5 features
        
        if data.empty or data.isnull().values.any() or len(data) < 200:
            print(f" Not enough valid data for {ticker}, skipping.")
            failed_tickers.append(ticker)
            continue
    except Exception as e:
        print(f" Error fetching data for {ticker}: {e}, skipping.")
        failed_tickers.append(ticker)
        continue

    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    
    X, y = create_sequences(scaled_data, WINDOW_SIZE, HORIZON_SIZE, TARGET_COLUMN_INDEX)
    
    if X.shape[0] == 0:
        print(f" Could not create sequences for {ticker}, skipping.")
        failed_tickers.append(ticker)
        continue
        
    
    n_features = X.shape[2]

    
    model = build_lstm_model(WINDOW_SIZE, n_features, HORIZON_SIZE)
    model.fit(X, y, epochs=50, batch_size=32, verbose=0)
    
    
    model_filename = f"{model_dir}/model_{ticker}.keras"
    scaler_filename = f"{model_dir}/scaler_{ticker}.pkl"
    
    model.save(model_filename)
    joblib.dump(scaler, scaler_filename) 
    
    stock_end_time = time.time()
    print(f" Done in {stock_end_time - stock_start_time:.2f}s. Saved.")
    successful_models += 1

end_total_time = time.time()

print("\n--- Multivariate LSTM Training Complete ---")
print(f"Total time: {(end_total_time - start_total_time) / 60:.2f} minutes")
print(f"Successfully trained models: {successful_models}")
print(f"Failed tickers: {len(failed_tickers)}")
if failed_tickers:
    print(f"Failed list: {', '.join(failed_tickers)}")