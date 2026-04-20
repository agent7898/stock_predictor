# Stock Price Trend Predictor

A full-stack web application that predicts stock price trends using Machine Learning (Linear Regression, ARIMA, LSTM).

## Startup Instructions

Run the following commands to start the application:

```bash
cd stock_predictor
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000` in your web browser.

**Note:** The first prediction run will take 60–120 seconds due to ARIMA grid search and LSTM training. Subsequent tickers are equally slow as the models retrain per request. This is expected behavior.
