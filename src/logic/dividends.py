import yfinance as yf

def get_dividend_forecast(symbol: str) -> dict:
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        current_price = ticker.history(period="1d")['Close'].iloc[-1]
        if 'dividendYield' not in info or not info['dividendYield'] or current_price == 0:
            return None

        hist_div = ticker.dividends
        if hist_div.empty:
            return None

        monthly_div = hist_div.resample('M').sum().fillna(0)
        avg_monthly_div = monthly_div[-12:].mean()
        annual_div_estimate = avg_monthly_div * 12
        predicted_yield = (annual_div_estimate / current_price) * 100

        last_payout = hist_div[-1] if not hist_div.empty else 0
        last_ex_div = hist_div.index[-1].strftime('%Y-%m-%d') if not hist_div.empty else 'N/A'

        return {
            'yield': round(predicted_yield, 2),
            'next_ex_date': last_ex_div,
            'annual_dividend': round(annual_div_estimate, 2),
            'recent_dividend': round(last_payout, 2)
        }
    except:
        return None
