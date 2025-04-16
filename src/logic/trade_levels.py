import pandas as pd

def calculate_trade_levels(data: pd.DataFrame, predicted_price: float) -> dict:
    close_prices = data['Close']
    current_price = close_prices.iloc[-1]
    sma_20 = close_prices.rolling(window=20).mean().iloc[-1]

    return {
        'buy': sma_20 * 0.98 if not pd.isna(sma_20) else current_price * 0.98,
        'entry': current_price,
        'stop_loss': current_price * 0.95
    }
