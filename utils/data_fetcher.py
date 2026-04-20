import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.download(ticker, period=period)
    if df.empty:
        raise ValueError(f"No historical data found for {ticker} over the '{period}' period.")
    
    df = df.dropna()
    
    if len(df) < 100:
        raise ValueError(f"Not enough historical trading days for '{ticker}' (found {len(df)}). We need at least 100 days of history to train the LSTM & ARIMA models. Please select a longer time period or choose an older stock.")
        
    df = df.reset_index()
    if 'Date' not in df.columns:
        raise ValueError("Date column missing from fetched data")
        
    df['Date'] = df['Date'].dt.strftime("%Y-%m-%d")
    return df[['Date', 'Open', 'High', 'Low', 'Close']]
