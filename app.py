import streamlit as st
import pandas as pd
from src.api.data_fetcher import fetch_stock_data
from src.ml.predictor import (
    predict_price, forecast_prices,
    create_forecast_chart, get_analyst_rating,
    adjust_levels_for_risk, show_predicted_dividend
)
from src.logic.trade_levels import calculate_trade_levels
from src.logic.arbitrage import analyze_arbitrage
from src.ml.sentiment import get_sentiment_score
from src.ml.ensemble import generate_final_signal
from src.logic.dividends import get_dividend_forecast

st.set_page_config(page_title="Stock Analyzer", layout="wide")

st.title("📈 Stock Analyzer App")
tab1, tab2 = st.tabs(["🔍 Symbol Analysis", "🔄 Arbitrage Analysis"])

with tab1:
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT)", value="AAPL")
    forecast_choice = st.selectbox("Prediction Horizon", ["1 Day", "1 Week", "1 Month", "1 Year"])
    forecast_map = {"1 Day": 1, "1 Week": 5, "1 Month": 22, "1 Year": 252}
    forecast_days = forecast_map[forecast_choice]
    risk_level = st.slider("Select Risk Level (1 = Low, 10 = High)", 1, 10, 5)

    if symbol:
        data = fetch_stock_data(symbol)
        if data is not None:
            st.line_chart(data['Close'])

            sentiment = get_sentiment_score(symbol)
            st.write(f"📣 Sentiment Score: {sentiment:+.2f}")

            model_outputs = {
                'gradient_boost': {'price': predict_price(data, forecast_days)[0], 'confidence': 0.7},
                'prophet': {'price': predict_price(data, forecast_days)[0] * 1.02, 'confidence': 0.75},
                'lstm': {'price': predict_price(data, forecast_days)[0] * 1.01, 'confidence': 0.65},
                'automl': {'price': predict_price(data, forecast_days)[0] * 0.99, 'confidence': 0.6},
                'online': {'price': predict_price(data, forecast_days)[0], 'confidence': 0.68},
            }

            analyst_rating = get_analyst_rating(symbol)
            st.write(f"📊 Analyst Rating: {analyst_rating}")

            signal = generate_final_signal(model_outputs, sentiment)
            raw_levels = calculate_trade_levels(data, signal['final_price_target'])
            levels = adjust_levels_for_risk(raw_levels, risk_level)

            st.subheader(f"Final Action: {signal['action']}")
            st.metric("Target Price", f"${signal['final_price_target']:.2f}")
            st.metric("Confidence", f"{signal['confidence']:.0%}")
            st.metric("Buy Target", f"${levels['buy']:.2f}")
            st.metric("Entry Price", f"${levels['entry']:.2f}")
            st.metric("Stop Loss", f"${levels['stop_loss']:.2f}")
            st.write("Reasons:", signal['reasons'])

            forecast_df = forecast_prices(data, forecast_days=forecast_days)
            forecast_df['Price Target'] = signal['final_price_target']

            fig = create_forecast_chart(data, forecast_df, entry=levels['entry'], stop=levels['stop_loss'], target=signal['final_price_target'])
            st.plotly_chart(fig, use_container_width=True)

            dividend = get_dividend_forecast(symbol)
            if dividend:
                st.success(f"💰 Estimated Annual Dividend Yield: {dividend['yield']:.2f}%")
                st.write(f"📅 Last Ex-Dividend Date: {dividend['next_ex_date']}")
                st.write(f"📈 Estimated Annual Payout: ${dividend['annual_dividend']:.2f}")
                st.write(f"💵 Most Recent Dividend: ${dividend['recent_dividend']:.2f}")
            else:
                st.info("No dividend information available for this symbol.")

            # Show predicted dividend
            show_predicted_dividend(symbol)

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
