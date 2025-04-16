def calculate_trade_levels(data, target_price):
    entry = data['Close'].iloc[-1]
    buffer = (target_price - entry) * 0.2

    return {
        'buy': entry - buffer,
        'entry': entry,
        'stop_loss': entry - buffer * 2
    }
