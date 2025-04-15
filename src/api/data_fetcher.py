import yfinance as yf
import pandas as pd

def fetch_stock_data(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        if hist.empty:
            return None
        return hist
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

