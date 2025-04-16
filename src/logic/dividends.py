import yfinance as yf

def get_dividend_forecast(symbol: str) -> dict:
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if 'dividendYield' in info and info['dividendYield']:
            next_ex = info.get('exDividendDate', None)
            if next_ex:
                import datetime
                next_ex = datetime.datetime.fromtimestamp(next_ex).strftime('%Y-%m-%d')
            return {
                'yield': info['dividendYield'] * 100,
                'next_ex_date': next_ex or 'N/A'
            }
        return None
    except:
        return None

