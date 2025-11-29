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
import random

# --- KEY FIX: Force a specific User-Agent globally ---
# This monkey-patch prevents Yahoo Finance from blocking us as a "bot".
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

# --- App Setup ---
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

def generate_dummy_data(ticker):
    """Generates realistic demo data if Yahoo blocks us."""
    print(f"⚠️ Yahoo blocked us. Generating demo data for {ticker}...")
    
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365)
    dates_index = pd.date_range(start=start_date, end=end_date)
    dates = dates_index.strftime('%Y-%m-%d').tolist()
    
    prices = [150.0]
    for _ in range(len(dates)-1):
        change = random.uniform(-2, 2.1)
        new_price = max(10, prices[-1] + change)
        prices.append(new_price)
        
    chart_data = {'dates': dates, 'prices': prices}
    
    # --- NEW: Current Price for Demo Mode ---
    current_price = round(prices[-1], 2)
    
    forecast_with_dates = []
    curr = current_price
    curr_date = end_date
    
    for i in range(HORIZON_SIZE):
        curr_date += datetime.timedelta(days=1)
        if curr_date.weekday() >= 5: curr_date += datetime.timedelta(days=2)
        curr += random.uniform(-1, 1)
        forecast_with_dates.append({'date': curr_date.strftime('%Y-%m-%d'), 'price': curr})
        
    return {
        'ticker': ticker,
        'companyName': f"{ticker} (Demo Mode - Live Data Blocked)",
        'currentPrice': current_price,
        'chartData': chart_data,
        'sevenDayForecast': forecast_with_dates
    }

@app.route('/')
def home():
    return "Stock Price Predictor API (V11 - Current Price Added) is running!"

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
        # Fetch live data
        ticker_obj = yf.Ticker(ticker_str)
        
        try:
            company_name = ticker_obj.info.get('longName', ticker_str)
        except:
            company_name = ticker_str

        hist_chart = ticker_obj.history(period='1y')
        if hist_chart.empty:
            # Fallback
            hist_chart = ticker_obj.history(period='2y')
            if hist_chart.empty:
                 raise ValueError("Empty data returned from Yahoo Finance")
            hist_chart = hist_chart.iloc[-252:]

        # --- NEW: Get Live Current Price ---
        current_price = round(hist_chart['Close'].iloc[-1], 2)

        hist_chart.reset_index(inplace=True)
        hist_chart['Date'] = hist_chart['Date'].dt.strftime('%Y-%m-%d')
        chart_data = {'dates': hist_chart['Date'].tolist(), 'prices': hist_chart['Close'].tolist()}
        
        # Prediction Data
        hist_pred = ticker_obj.history(period='100d')
        model_features = hist_pred[FEATURES]
        
        if len(model_features) < WINDOW_SIZE:
             raise ValueError("Not enough historical data")

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
            'currentPrice': current_price,
            'chartData': chart_data,
            'sevenDayForecast': forecast_with_dates
        })

    except Exception as e:
        print(f"Live fetch failed for {ticker_str}: {e}")
        return jsonify(generate_dummy_data(ticker_str))

if __name__ == '__main__':
    app.run(debug=True, port=5000)