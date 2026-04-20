import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.download(ticker, period=period)
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    
    df = df.dropna()
    df = df.reset_index()
    if 'Date' not in df.columns:
        raise ValueError("Date column missing from fetched data")
        
    df['Date'] = df['Date'].dt.strftime("%Y-%m-%d")
    return df[['Date', 'Open', 'High', 'Low', 'Close']]
