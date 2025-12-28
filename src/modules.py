import yfinance as yf
import pandas as pd
import numpy as np
from newsapi import NewsApiClient
from transformers import pipeline
import streamlit as st

class DataPipeline:
    def __init__(self, news_api_key):
        self.news_api = NewsApiClient(api_key=news_api_key)
        # Cache the sensitive FinBERT model to avoid reloading
        self.sentiment_pipe = pipeline("text-classification", model="ProsusAI/finbert")

    def fetch_stock_data(self, ticker, period="1y"):
        """Fetches OHLCV data from Yahoo Finance"""
        df = yf.download(ticker, period=period)
        
        # --- FIX: FLATTEN MULTI-INDEX COLUMNS ---
        # yfinance often returns columns like ('Close', 'AAPL'). We want just 'Close'.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # ----------------------------------------

        df.reset_index(inplace=True)
        return df

    def add_technical_indicators(self, df):
        """Calculates RSI, MACD, EMA (Feature Engineering)"""
        # Ensure Close is 1D
        close = df['Close'].squeeze()

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # EMA
        df['EMA_20'] = close.ewm(span=20, adjust=False).mean()

        # MACD
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        
        df.dropna(inplace=True)
        return df

    def fetch_and_analyze_news(self, ticker, days=29):
        """
        Fetches news and calculates Sentiment.
        Returns a DataFrame with Date and Sentiment_Score.
        """
        try:
            # Note: NewsAPI free tier strictly limits historical access
            all_articles = self.news_api.get_everything(q=ticker,
                                                      language='en',
                                                      sort_by='relevancy',
                                                      page_size=50)
            
            sentiments = []
            dates = []

            if not all_articles.get('articles'):
                return pd.DataFrame(columns=['Date', 'Sentiment'])

            for article in all_articles['articles']:
                title = article['title']
                pub_date = article['publishedAt'][:10] # YYYY-MM-DD
                
                if title: # Ensure title is not None
                    # FinBERT Prediction
                    result = self.sentiment_pipe(title)[0]
                    score = result['score']
                    if result['label'] == 'negative': score *= -1
                    elif result['label'] == 'neutral': score = 0
                    
                    sentiments.append(score)
                    dates.append(pub_date)

            news_df = pd.DataFrame({'Date': dates, 'Sentiment': sentiments})
            news_df['Date'] = pd.to_datetime(news_df['Date'])
            
            # Aggregate sentiment by day (mean score)
            daily_sentiment = news_df.groupby('Date').mean().reset_index()
            return daily_sentiment

        except Exception as e:
            st.error(f"Error fetching news: {e}")
            return pd.DataFrame(columns=['Date', 'Sentiment'])
