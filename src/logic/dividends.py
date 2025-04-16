import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def get_dividend_forecast(symbol: str) -> dict:
    try:
        ticker = yf.Ticker(symbol)
        hist_div = ticker.dividends
        current_price = ticker.history(period="1d")['Close'].iloc[-1]

        if hist_div.empty or current_price == 0:
            return None

        # Resample to monthly, fill missing with 0
        monthly_div = hist_div.resample('M').sum().fillna(0)

        # Calculate average monthly payout and annualize
        avg_monthly_div = monthly_div[-12:].mean()
        annual_div_estimate = avg_monthly_div * 12

        # Estimate yield
        predicted_yield = (annual_div_estimate / current_price) * 100

        # Try to extract most recent dividend and ex-div date
        last_payout = hist_div[-1] if not hist_div.empty else 0
        last_ex_div = hist_div.index[-1].strftime('%Y-%m-%d') if not hist_div.empty else 'N/A'

        return {
            'yield': round(predicted_yield, 2),
            'next_ex_date': last_ex_div,
            'annual_dividend': round(annual_div_estimate, 2),
            'recent_dividend': round(last_payout, 2)
        }

    except Exception as e:
        print(f"Dividend error for {symbol}: {e}")
        return None
