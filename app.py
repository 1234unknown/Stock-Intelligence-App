import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath("src"))


from src.api.data_fetcher import fetch_stock_data
from src.ml.predictor import (
    predict_price, forecast_prices,
    create_forecast_chart, get_analyst_rating,
    adjust_levels_for_risk, show_predicted_dividend,
    scan_short_term_trades
)
from src.logic.trade_levels import calculate_trade_levels
from src.logic.arbitrage import analyze_arbitrage
from src.ml.sentiment import get_sentiment_score
from src.ml.ensemble import generate_final_signal
from src.logic.dividends import get_dividend_forecast

st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📈 Stock Analyzer App")

tab1, tab2, tab3 = st.tabs(["🔍 Symbol Analysis", "🔄 Arbitrage Analysis", "🧠 Options Scanner"])


# ---------------------- Option Suggestion Logic ----------------------
def suggest_option_trade(symbol, current_price, predicted_price, forecast_days):
    try:
        ticker = yf.Ticker(symbol)
        expiry_dates = ticker.options
        if not expiry_dates:
            return None

        today = datetime.now().date()
        expiry = next((d for d in expiry_dates if datetime.strptime(d, "%Y-%m-%d").date() > today + timedelta(days=forecast_days)), expiry_dates[0])
        option_chain = ticker.option_chain(expiry)

        option_type = "CALL" if predicted_price > current_price else "PUT"
        options_df = option_chain.calls if option_type == "CALL" else option_chain.puts
        options_df = options_df.sort_values(by='strike')

        if option_type == "CALL":
            row = options_df[options_df['strike'] >= predicted_price].head(1)
        else:
            row = options_df[options_df['strike'] <= predicted_price].tail(1)

        if row.empty:
            return None

        return {
            "Strike": round(row['strike'].values[0], 2),
            "Expiration": expiry,
            "Option Type": option_type,
            "Delta": row['delta'].values[0] if 'delta' in row else None,
            "IV": row['impliedVolatility'].values[0] if 'impliedVolatility' in row else None
        }
    except Exception as e:
        print(f"Option lookup failed for {symbol}: {e}")
        return None


# ---------------------- Tab 1: Symbol Analysis ----------------------
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

            show_predicted_dividend(symbol)

            option = suggest_option_trade(symbol, levels['entry'], signal['final_price_target'], forecast_days)
            if option:
                st.info(f"📌 Suggested {option['Option Type']} Option: Strike ${option['Strike']} expiring {option['Expiration']}")
        else:
            st.error("Failed to fetch data. Check the symbol and try again.")


# ---------------------- Tab 2: Arbitrage ----------------------
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


# ---------------------- Tab 3: Options Scanner ----------------------
with tab3:
    st.title("🧠 Options Scanner")
    tickers = st.text_input("Enter comma-separated stock tickers to scan", "AAPL,MSFT,TSLA,NVDA,AMD,GOOGL,META,NFLX")
    tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    signal_filter = st.selectbox("Filter by signal", ["ALL", "BUY", "WATCH"])
    export_csv = st.checkbox("Enable CSV Export")

    rows = []
    for symbol in tickers_list:
        try:
            data = fetch_stock_data(symbol)
            if data is None or len(data) < 30:
                continue

            current_price = data['Close'].iloc[-1]
            forecast_days = 5
            predicted_price, _ = predict_price(data, forecast_days)
            delta_pct = ((predicted_price - current_price) / current_price) * 100

            option = suggest_option_trade(symbol, current_price, predicted_price, forecast_days)
            greek_score = 0
            if option and option['Delta'] and option['IV']:
                delta = option['Delta']
                iv = option['IV']
                delta_score = 1 - abs(0.5 - delta) * 2
                iv_penalty = min(iv, 1.0)
                greek_score = delta_score * 0.7 - iv_penalty * 0.3

            signal_strength = delta_pct + (greek_score * 100)
            signal = "BUY" if signal_strength > 5 else "WATCH"

            if signal_filter != "ALL" and signal != signal_filter:
                continue

            rows.append({
                "Symbol": symbol,
                "Current Price": round(current_price, 2),
                "Predicted Price": round(predicted_price, 2),
                "Δ %": round(delta_pct, 2),
                "Signal": signal,
                "Strike": option['Strike'] if option else "-",
                "Expiry": option['Expiration'] if option else "-",
                "Option Type": option['Option Type'] if option else "-",
                "Delta": round(option['Delta'], 2) if option and option['Delta'] else "-",
                "IV": round(option['IV'] * 100, 2) if option and option['IV'] else "-",
                "Greek Score": round(greek_score, 2) if greek_score else "-",
                "Signal Strength": round(signal_strength, 2)
            })

            fig, ax = plt.subplots(figsize=(3, 1))
            ax.plot(data['Close'][-30:], linewidth=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(symbol, fontsize=8)
            st.pyplot(fig)

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values(by=["Signal Strength"], ascending=False)
        st.dataframe(df)

        st.subheader("📊 Signal Strength Overview")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(df['Symbol'], df['Signal Strength'], color="skyblue")
        ax.set_ylabel("Signal Strength")
        ax.set_title("Top Trade Signals by Combined ML + Greeks")
        st.pyplot(fig)

        if export_csv:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "option_signals.csv", "text/csv")
    else:
        st.warning("No valid data returned from scan.")
