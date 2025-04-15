from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd

def predict_price(data: pd.DataFrame) -> tuple:
    data = data.dropna()
    if len(data) < 30:
        return data['Close'].iloc[-1], 0.5

    X = np.arange(len(data)).reshape(-1, 1)
    y = data['Close'].values
    model = GradientBoostingRegressor().fit(X, y)
    next_day = np.array([[len(data)]])
    predicted = model.predict(next_day)
    confidence = model.score(X, y)
    return predicted[0], confidence

def forecast_prices(data: pd.DataFrame, forecast_days: int = 5) -> pd.DataFrame:
    data = data.dropna()
    if len(data) < 30:
        last_price = data['Close'].iloc[-1]
        return pd.DataFrame({'Forecast': [last_price] * forecast_days})

    X = np.arange(len(data)).reshape(-1, 1)
    y = data['Close'].values
    model = GradientBoostingRegressor().fit(X, y)
    future_X = np.arange(len(data), len(data) + forecast_days).reshape(-1, 1)
    future_y = model.predict(future_X)

    forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')
    forecast_df = pd.DataFrame({'Forecast': future_y}, index=forecast_index)
    return forecast_df
