from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd

def predict_price(data: pd.DataFrame, forecast_days: int = 1) -> tuple:
    data = data.dropna()
    if len(data) < 30:
        return data['Close'].iloc[-1], 0.5

    X = np.arange(len(data)).reshape(-1, 1)
    y = data['Close'].values
    model = GradientBoostingRegressor().fit(X, y)
    future_day = np.array([[len(data) + forecast_days - 1]])
    predicted = model.predict(future_day)
    confidence = model.score(X, y)
    return predicted[0], confidence
