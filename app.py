import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import yfinance as yf
from utils.data_fetcher import fetch_stock_data
from utils.sentiment import compute_sentiment
from models.linear_model import LinearRegressionModel
from models.arima_model import ArimaModel
from models.lstm_model import LSTMModel

app = Flask(__name__)

def convert_numpy(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_numpy(i) for i in obj]
    return obj

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        ticker = data.get('ticker')
        period = data.get('period', '2y')
        forecast_days = int(data.get('forecast_days', 30))
        
        df = fetch_stock_data(ticker, period)
        prices = df['Close'].values.astype(np.float64).flatten()
        
        if np.isnan(prices).any():
            raise ValueError("Dataset contains NaN values after preprocessing.")
        
        lr_results = LinearRegressionModel().train_and_predict(prices, forecast_days)
        arima_results = ArimaModel().train_and_predict(prices, forecast_days)
        lstm_results = LSTMModel().train_and_predict(prices, forecast_days)

        # Use one strict common test window so every model is compared on
        # exactly the same number of points and the same date range.
        common_len = min(
            len(lr_results["test_actual"]),
            len(arima_results["test_actual"]),
            len(lstm_results["test_actual"])
        )
        if common_len <= 0:
            raise ValueError("Insufficient test predictions to compare models.")

        lr_actual = lr_results["test_actual"][-common_len:]
        lr_pred = lr_results["test_predicted"][-common_len:]
        arima_actual = arima_results["test_actual"][-common_len:]
        arima_pred = arima_results["test_predicted"][-common_len:]
        lstm_actual = lstm_results["test_actual"][-common_len:]
        lstm_pred = lstm_results["test_predicted"][-common_len:]

        test_dates = df["Date"].tolist()[-common_len:]
        
        last_date = df['Date'].iloc[-1]
        forecast_dates = pd.date_range(start=pd.to_datetime(last_date) + pd.Timedelta(days=1), periods=forecast_days, freq='B')
        forecast_dates_str = forecast_dates.strftime("%Y-%m-%d").tolist()
        
        response = {
            "ticker": ticker,
            "dates": df['Date'].tolist(),
            "open": df['Open'].values.astype(np.float64).flatten().tolist(),
            "high": df['High'].values.astype(np.float64).flatten().tolist(),
            "low": df['Low'].values.astype(np.float64).flatten().tolist(),
            "prices": prices.tolist(),
            "test_dates": test_dates,
            "forecast_dates": forecast_dates_str,
            "linear_regression": {
                **lr_results,
                "test_actual": lr_actual,
                "test_predicted": lr_pred
            },
            "arima": {
                **arima_results,
                "test_actual": arima_actual,
                "test_predicted": arima_pred
            },
            "lstm": {
                **lstm_results,
                "test_actual": lstm_actual,
                "test_predicted": lstm_pred
            }
        }
        
        return jsonify(convert_numpy(response))
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", 
               "META", "NFLX", "AMD", "RELIANCE.NS", "TCS.NS", "INFY.NS"]
    stocks = []
    
    for ticker in tickers:
        try:
            tkr = yf.Ticker(ticker)
            info = tkr.fast_info
            last_price = info.last_price
            prev_close = info.previous_close
            change = last_price - prev_close
            change_pct = (change / prev_close) * 100
            
            if change > 0:
                direction = "up"
            elif change < 0:
                direction = "down"
            else:
                direction = "flat"
                
            stocks.append({
                "ticker": ticker,
                "price": round(float(last_price), 2),
                "change": round(float(change), 2),
                "change_pct": round(float(change_pct), 2),
                "direction": direction
            })
        except Exception:
            # Skip silently on error as required
            continue
            
    return jsonify(convert_numpy({"stocks": stocks}))

@app.route('/api/sentiment/<ticker>', methods=['GET'])
def get_sentiment(ticker):
    try:
        result = compute_sentiment(ticker)
        return jsonify(convert_numpy(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
