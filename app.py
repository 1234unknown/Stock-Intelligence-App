import streamlit as st
import pandas as pd
from src.api.data_fetcher import fetch_stock_data
from src.ml.predictor import predict_price, forecast_prices
from src.logic.trade_levels import calculate_trade_levels
from src.logic.arbitrage import analyze_arbitrage

st.set_page_config(page_title="Stock Analyzer", layout="wide")

st.title("📈 Stock Analyzer App")
tab1, tab2 = st.tabs(["🔍 Symbol Analysis", "🔄 Arbitrage Analysis"])

with tab1:
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT)", value="AAPL")

    if symbol:
        data = fetch_stock_data(symbol)
        if data is not None:
            st.line_chart(data['Close'])

            prediction, confidence = predict_price(data)
            st.subheader(f"Predicted Price: ${prediction:.2f} (Confidence: {confidence:.1%})")

            levels = calculate_trade_levels(data, prediction)
            st.metric("Action", levels['action'])
            st.metric("Buy Target", f"${levels['buy']:.2f}")
            st.metric("Entry Target", f"${levels['entry']:.2f}")
            st.metric("Stop Loss", f"${levels['stop_loss']:.2f}")

            forecast_df = forecast_prices(data, forecast_days=5)
            forecast_df['Entry Target'] = levels['entry']
            forecast_df['Stop Loss'] = levels['stop_loss']
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

