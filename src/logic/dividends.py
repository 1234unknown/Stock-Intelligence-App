import yfinance as yf

def get_dividend_forecast(symbol: str):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        dividend_yield = info.get('dividendYield', 0.0)
        annual_dividend = info.get('dividendRate', 0.0)
        ex_date = info.get('exDividendDate', None)
        recent_div = ticker.dividends[-1] if not ticker.dividends.empty else 0.0

        return {
            'yield': dividend_yield * 100 if dividend_yield else 0.0,
            'annual_dividend': annual_dividend,
            'next_ex_date': ex_date,
            'recent_dividend': recent_div
        }
    except Exception as e:
        print(f"Error fetching dividend data for {symbol}: {e}")
        return None
