from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import streamlit as st

def predict_price(data: pd.DataFrame, forecast_days: int = 1) -> tuple:
    data = data.dropna()
    if len(data) < 30 + forecast_days:
        return data['Close'].iloc[-1], 0.5

    lookback = 30
    X, y = [], []
    for i in range(len(data) - lookback - forecast_days):
        X.append(data['Close'].iloc[i:i + lookback].values)
        y.append(data['Close'].iloc[i + lookback + forecast_days - 1])

    X, y = np.array(X), np.array(y)
    model = GradientBoostingRegressor().fit(X, y)
    recent_data = data['Close'].iloc[-lookback:].values.reshape(1, -1)
    prediction = model.predict(recent_data)
    confidence = model.score(X, y)
    return prediction[0], confidence

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
    return pd.DataFrame({'Forecast': future_y}, index=forecast_index)

def create_forecast_chart(data: pd.DataFrame, forecast_df: pd.DataFrame, entry: float, stop: float, target: float):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical'))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines+markers', name='Forecast'))
    fig.add_hline(y=entry, line=dict(dash='dot'), annotation_text="Entry", annotation_position="top left")
    fig.add_hline(y=stop, line=dict(color='red', dash='dot'), annotation_text="Stop Loss", annotation_position="bottom left")
    fig.add_hline(y=target, line=dict(color='green', dash='dot'), annotation_text="Target", annotation_position="top right")
    fig.update_layout(title="Forecast vs Historical with Trade Levels", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
    return fig

def get_analyst_rating(symbol: str) -> str:
    try:
        info = yf.Ticker(symbol).info
        return info.get("recommendationKey", "none").upper()
    except:
        return "NONE"

def adjust_levels_for_risk(levels: dict, risk_level: int) -> dict:
    multiplier = 1 + (risk_level - 5) * 0.05
    return {
        'buy': levels['buy'] * multiplier,
        'entry': levels['entry'],
        'stop_loss': levels['stop_loss'] / multiplier
    }

def predict_next_dividend(symbol: str) -> float:
    try:
        ticker = yf.Ticker(symbol)
        div = ticker.dividends
        if len(div) < 3:
            return 0.0
        div = div[-12:].resample('M').sum().fillna(0)
        X = np.arange(len(div)).reshape(-1, 1)
        y = div.values
        model = RandomForestRegressor().fit(X, y)
        next_div = model.predict(np.array([[len(div)]]))
        return round(next_div[0], 4)
    except Exception as e:
        print(f"Dividend prediction failed for {symbol}: {e}")
        return 0.0

def show_predicted_dividend(symbol: str):
    predicted_div = predict_next_dividend(symbol)
    if predicted_div > 0:
        st.info(f"🔮 Predicted Next Dividend Amount: ${predicted_div:.4f}")
