from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import os
import datetime
import requests

# --- KEY FIX: Force a specific User-Agent globally ---
# This tells Yahoo we are a specific Chrome browser, avoiding the "Bot" block.
yf.pdr_override()

# Monkey-patch the requests library to always send this header
old_get = requests.get
def new_get(*args, **kwargs):
    headers = kwargs.get('headers', {})
    headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    kwargs['headers'] = headers
    return old_get(*args, **kwargs)
requests.get = new_get

# --- Constants ---
WINDOW_SIZE = 60
HORIZON_SIZE = 7
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
TARGET_COLUMN_INDEX = 3

app = Flask(__name__)
CORS(app)
MODEL_DIR = "models"
loaded_models_cache = {}

def get_model_and_scaler(ticker):
    if ticker in loaded_models_cache:
        return loaded_models_cache[ticker]
    
    model_path = os.path.join(MODEL_DIR, f"model_{ticker}.keras")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{ticker}.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
        
    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        loaded_models_cache[ticker] = (model, scaler)
        return model, scaler
    except Exception as e:
        print(f"Error loading model for {ticker}: {e}")
        return None, None

@app.route('/')
def home():
    return "Stock Price Predictor API (V8 - User-Agent Patch) is running!"

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    ticker_str = data.get('ticker')
    if not ticker_str:
        return jsonify({'error': 'No ticker provided.'}), 400
    
    ticker_str = ticker_str.upper()
    
    model, scaler = get_model_and_scaler(ticker_str)
    if model is None:
        return jsonify({'error': f'No pre-trained model available for {ticker_str}.'}), 404

    try:
        # Fetch data (The monkey-patch above handles the headers now)
        ticker_obj = yf.Ticker(ticker_str)
        
        company_name = ticker_obj.info.get('longName', ticker_str)
        
        # Get 1 year of data
        hist_chart = ticker_obj.history(period='1y')
        if hist_chart.empty:
            # If 1y fails, try fetching max and slicing
            hist_chart = ticker_obj.history(period='2y')
            if hist_chart.empty:
                return jsonify({'error': 'Yahoo Finance blocked the request (Rate Limit). Please try again in 1 minute.'}), 429
            hist_chart = hist_chart.iloc[-252:] # Approx 1 year

        hist_chart.reset_index(inplace=True)
        hist_chart['Date'] = hist_chart['Date'].dt.strftime('%Y-%m-%d')
        chart_data = {'dates': hist_chart['Date'].tolist(), 'prices': hist_chart['Close'].tolist()}
        
        # Get data for prediction
        # We reuse the data we just fetched if possible to save requests
        model_features = hist_chart[FEATURES]
        
        if len(model_features) < WINDOW_SIZE:
             return jsonify({'error': 'Not enough historical data to predict.'}), 400

        last_known_date = model_features.index[-1].date()
        last_60_days = model_features.iloc[-WINDOW_SIZE:]
        
        scaled_input = scaler.transform(last_60_days)
        input_data = np.reshape(scaled_input, (1, WINDOW_SIZE, len(FEATURES)))
        scaled_prediction = model.predict(input_data)
        
        dummy_array = np.zeros((HORIZON_SIZE, len(FEATURES)))
        dummy_array[:, TARGET_COLUMN_INDEX] = scaled_prediction[0]
        unscaled_prediction_array = scaler.inverse_transform(dummy_array)
        seven_day_forecast_prices = unscaled_prediction_array[:, TARGET_COLUMN_INDEX]
        seven_day_forecast_list = seven_day_forecast_prices.tolist()

        forecast_dates = []
        current_date = last_known_date
        while len(forecast_dates) < HORIZON_SIZE:
            current_date += datetime.timedelta(days=1)
            if current_date.weekday() < 5: 
                forecast_dates.append(current_date.strftime('%Y-%m-%d'))
        
        forecast_with_dates = []
        for i in range(HORIZON_SIZE):
            forecast_with_dates.append({
                'date': forecast_dates[i],
                'price': seven_day_forecast_list[i]
            })

        return jsonify({
            'ticker': ticker_str,
            'companyName': company_name,
            'chartData': chart_data,
            'sevenDayForecast': forecast_with_dates
        })

    except Exception as e:
        print(f"FULL ERROR for {ticker_str}: {e}")
        return jsonify({'error': f'Backend Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)