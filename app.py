import streamlit as st
import pandas as pd
from src.api.data_fetcher import fetch_stock_data
from src.ml.predictor import predict_price, forecast_prices
from src.logic.trade_levels import calculate_trade_levels
from src.logic.arbitrage import analyze_arbitrage
from src.ml.sentiment import get_sentiment_score
from src.ml.ensemble import generate_final_signal

st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📈 Stock Analyzer App")

tab1, tab2 = st.tabs(["🔍 Symbol Analysis", "🔄 Arbitrage Analysis"])

with tab1:
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT)", value="AAPL")

    if symbol:
        data = fetch_stock_data(symbol)
        if data is not None:
            st.line_chart(data['Close'])

            sentiment = get_sentiment_score(symbol)
            st.write(f"📣 Sentiment Score: {sentiment:+.2f}")

            model_outputs = {
                'gradient_boost': {'price': predict_price(data)[0], 'confidence': 0.7},
                'prophet': {'price': predict_price(data)[0] * 1.02, 'confidence': 0.75},
                'lstm': {'price': predict_price(data)[0] * 1.01, 'confidence': 0.65},
                'automl': {'price': predict_price(data)[0] * 0.99, 'confidence': 0.6},
                'online': {'price': predict_price(data)[0], 'confidence': 0.68},
            }

            signal = generate_final_signal(model_outputs, sentiment)

            st.subheader(f"Final Action: {signal['action']}")
            st.metric("Target Price", f"${signal['final_price_target']:.2f}")
            st.metric("Confidence", f"{signal['confidence']:.0%}")
            st.write("Reasons:", signal['reasons'])

            forecast_df = forecast_prices(data, forecast_days=5)
            forecast_df['Price Target'] = signal['final_price_target']
            st.line_chart(forecast_df)
        else:
            st.error("Failed to fetch data. Check the symbol and try again.")

with tab2:
    st.subheader("Arbitrage Opportunity Finder")
    symbol1 = st.text_input("Symbol 1", value="AAPL")
    symbol2 = st.text_input("Symbol 2", value="MSFT")

    if symbol1 and symbol2:
        data1 = fetch_stock_data(symbol1)
        data2 = fetch_stock_data(symbol2)

        if data1 is not None and data2 is not None:
            result = analyze_arbitrage(data1, data2)
            st.line_chart(result['spread'])
            st.write("Correlation:", result['correlation'])
            st.write("Z-score (last):", result['z_score'][-1])

            z_alert = "No significant signal"
            if result['z_score'][-1] > 2:
                z_alert = f"📈 Z-score {result['z_score'][-1]:.2f} > 2: Consider Short {symbol1} / Long {symbol2}"
            elif result['z_score'][-1] < -2:
                z_alert = f"📉 Z-score {result['z_score'][-1]:.2f} < -2: Consider Long {symbol1} / Short {symbol2}"

            st.warning(z_alert)
        else:
            st.error("Could not fetch data for one or both symbols.")
