import pandas as pd
import numpy as np
from scipy.stats import zscore

def analyze_arbitrage(data1: pd.DataFrame, data2: pd.DataFrame) -> dict:
    merged = pd.merge(data1['Close'], data2['Close'], left_index=True, right_index=True, suffixes=("_1", "_2"))
    merged.dropna(inplace=True)

    spread = merged['Close_1'] - merged['Close_2']
    corr = merged['Close_1'].corr(merged['Close_2'])
    z_scores = zscore(spread)

    result = {
        'spread': spread,
        'correlation': round(corr, 4),
        'z_score': z_scores
    }
    return result
