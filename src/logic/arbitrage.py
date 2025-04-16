import pandas as pd
import numpy as np

def analyze_arbitrage(data1, data2):
    min_len = min(len(data1), len(data2))
    spread = data1['Close'].iloc[-min_len:] - data2['Close'].iloc[-min_len:]
    z_score = (spread - spread.mean()) / spread.std()
    return {
        'spread': spread,
        'z_score': z_score,
        'correlation': data1['Close'].iloc[-min_len:].corr(data2['Close'].iloc[-min_len:])
    }
