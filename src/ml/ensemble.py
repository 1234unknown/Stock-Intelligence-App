def generate_final_signal(models_output: dict, sentiment: float) -> dict:
    total_weight = sum(m['confidence'] for m in models_output.values())
    weighted_price = sum(m['price'] * m['confidence'] for m in models_output.values()) / total_weight
    avg_conf = total_weight / len(models_output)

    sentiment_boost = 1 + (sentiment * 0.05)
    adjusted_price = weighted_price * sentiment_boost

    base_price = models_output['gradient_boost']['price']
    delta = adjusted_price - base_price

    if delta > base_price * 0.03:
        action = 'BUY'
    elif delta < -base_price * 0.03:
        action = 'SELL'
    else:
        action = 'HOLD'

    return {
        'final_price_target': adjusted_price,
        'action': action,
        'confidence': avg_conf,
        'reasons': [
            f"Model avg target: ${weighted_price:.2f}",
            f"Sentiment factor applied: {sentiment:+.2f}",
            f"Delta from base: ${delta:.2f}"
        ]
    }

