from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from utils.data_fetcher import fetch_stock_data
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
        
        lr_results = LinearRegressionModel().train_and_predict(prices, forecast_days)
        arima_results = ArimaModel().train_and_predict(prices, forecast_days)
        lstm_results = LSTMModel().train_and_predict(prices, forecast_days)
        
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
            "forecast_dates": forecast_dates_str,
            "linear_regression": lr_results,
            "arima": arima_results,
            "lstm": lstm_results
        }
        
        return jsonify(convert_numpy(response))
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
