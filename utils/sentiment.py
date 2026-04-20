import yfinance as yf
import numpy as np
import pandas as pd

def compute_sentiment(ticker: str) -> dict:
    df = yf.download(ticker, period="3mo", progress=False)
    if df.empty or len(df) < 50:
        raise ValueError(f"Not enough data to compute sentiment for {ticker}")
    
    close = df['Close'].values.astype(np.float64).flatten()
    
    # 1. RSI (14-day)
    deltas = np.diff(close)
    seed_deltas = deltas[-14:]
    gains = seed_deltas[seed_deltas > 0]
    losses = -seed_deltas[seed_deltas < 0]
    avg_gain = np.sum(gains) / 14 if len(gains) > 0 else 0
    avg_loss = np.sum(losses) / 14 if len(losses) > 0 else 0
    
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1 + rs))
        
    # 2. 20-day SMA vs 50-day SMA
    sma20 = np.mean(close[-20:])
    sma50 = np.mean(close[-50:])
    sma_signal = "bullish" if sma20 > sma50 else "bearish"
    
    # 3. Price vs 52-week high
    df_1y = yf.download(ticker, period="1y", progress=False)
    if not df_1y.empty:
        close_1y = df_1y['Close'].values.astype(np.float64).flatten()
        high_52w = np.max(close_1y[-252:])
        current = close[-1]
        pct_from_high = ((current - high_52w) / high_52w) * 100
    else:
        pct_from_high = 0.0
        
    # 4. 10-day momentum
    momentum = ((close[-1] - close[-10]) / close[-10]) * 100
    
    # 5. Volatility (20-day)
    returns = deltas[-20:] / close[-21:-1]
    volatility = np.std(returns) * np.sqrt(252) * 100
    if volatility < 20:
        vol_level = "low"
    elif volatility <= 40:
        vol_level = "moderate"
    else:
        vol_level = "high"
        
    # Scoring
    score = 0
    if rsi < 30: score += 2
    elif rsi > 70: score -= 2
    elif rsi > 50: score += 1
    
    if sma20 > sma50: score += 1
    else: score -= 1
    
    if momentum > 5: score += 1
    elif momentum < -5: score -= 1
    
    if pct_from_high > -5: score += 1
    
    # Overall sentiment label
    if score >= 3:
        sentiment = "Strongly Bullish"
        summary = f"{ticker} is showing extremely strong technical momentum with robust moving averages."
    elif score == 2:
        sentiment = "Bullish"
        summary = f"{ticker} is trending positively and remains well supported by recent price action."
    elif score == 1:
        sentiment = "Mildly Bullish"
        summary = f"{ticker} exhibits a slight upward bias but lacks strong conviction."
    elif score == 0:
        sentiment = "Neutral"
        summary = f"{ticker} is currently trading sideways with mixed technical signals."
    elif score == -1:
        sentiment = "Mildly Bearish"
        summary = f"{ticker} shows a slight downward bias with cautious technical indicators."
    elif score == -2:
        sentiment = "Bearish"
        summary = f"{ticker} is trending negatively, facing resistance across short-term moving averages."
    else:
        sentiment = "Strongly Bearish"
        summary = f"{ticker} is under severe selling pressure with very weak technicals."
        
    return {
        "ticker": ticker,
        "sentiment": sentiment,
        "score": int(score),
        "rsi": float(rsi),
        "sma_signal": sma_signal,
        "momentum_pct": float(momentum),
        "volatility": float(volatility),
        "volatility_level": vol_level,
        "pct_from_52w_high": float(pct_from_high),
        "summary": summary
    }
