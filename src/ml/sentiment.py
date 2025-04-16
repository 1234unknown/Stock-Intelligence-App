import requests
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon', quiet=True)

analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(symbol: str) -> float:
    api_key = st.secrets["api"]["finnhub_key"]
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from=2024-01-01&to=2024-12-31&token={api_key}"

    try:
        response = requests.get(url)
        news = response.json()
        headlines = [item['headline'] for item in news[:10]]
        if not headlines:
            return 0.0
        scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
        return sum(scores) / len(scores)
    except:
        return 0.0
