import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.modules import DataPipeline
from src.hybrid_model import HybridPredictor

# --- CONFIGURATION ---
st.set_page_config(page_title="FinSentStock", layout="wide")
st.title("📈 FinSentStock: News Sentiment + Technical Analysis")
st.markdown("predict short-term stock movements using **FinBERT Sentiment** and **Hybrid LSTM-LightGBM**.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Stock Ticker", value="AAPL")
    # Retrieve API Key from Streamlit Secrets (for Cloud) or User Input (for Local)
    try:
        api_key = st.secrets["NEWS_API_KEY"]
    except:
        api_key = st.text_input("Enter NewsAPI Key", type="password")
    
    st.info("Note: Free NewsAPI only supports data from the last 30 days.")

# --- MAIN LOGIC ---
if st.button("Run Analysis"):
    if not api_key:
        st.error("Please provide a NewsAPI Key to proceed.")
    else:
        with st.spinner('Initializing Pipeline...'):
            pipeline = DataPipeline(api_key)
            model = HybridPredictor(alpha=0.5)

        # 1. FETCH DATA
        with st.spinner(f'Fetching data for {ticker}...'):
            # Fetch Stock Data
            df_stock = pipeline.fetch_stock_data(ticker)
            df_stock = pipeline.add_technical_indicators(df_stock)
            
            # Fetch News Data
            df_news = pipeline.fetch_and_analyze_news(ticker)
        
        # 2. MERGE DATA
        # Merge on Date (inner join to keep only matching days)
        # Note: YFinance Date is Timestamp, News Date is object, need conversion
        df_stock['Date'] = pd.to_datetime(df_stock['Date']).dt.tz_localize(None)
        
        # Merge or just use Stock Data if no news found (robustness)
        if not df_news.empty:
            df_final = pd.merge(df_stock, df_news, on='Date', how='left')
            df_final['Sentiment'] = df_final['Sentiment'].fillna(0) # Fill missing sentiment with Neutral
        else:
            st.warning("No news found or API limit reached. Using Technicals only.")
            df_final = df_stock

        # Display Data
        col1, col2 = st.columns(2)
        col1.subheader("Price History")
        col1.line_chart(df_final.set_index('Date')['Close'])
        
        if 'Sentiment' in df_final.columns:
            col2.subheader("Sentiment Score (Daily Avg)")
            col2.bar_chart(df_final.set_index('Date')['Sentiment'])

        # 3. TRAIN MODEL
        with st.spinner('Training Hybrid Model (LSTM + LightGBM)...'):
            X, y = model.prepare_data(df_final)
            rmse, preds, actuals = model.train(X, y)
            
        st.success(f"Model Trained! Test RMSE: {rmse:.4f}")

        # 4. VISUALIZE PREDICTIONS (Test Set)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=actuals, mode='lines', name='Actual Price'))
        fig.add_trace(go.Scatter(y=preds, mode='lines', name='Predicted Price', line=dict(dash='dash')))
        fig.update_layout(title="Model Validation (Test Set)", xaxis_title="Days", yaxis_title="Price")
        st.plotly_chart(fig)

        # 5. FORECAST TOMORROW
        last_features = df_final[['Close', 'RSI', 'MACD', 'EMA_20', 'Sentiment']].iloc[-1].values
        next_price = model.predict_next(last_features)
        
        current_price = df_final['Close'].iloc[-1]
        change = next_price - current_price
        color = "green" if change > 0 else "red"
        
        st.metric(label="Predicted Next Day Price", value=f"${next_price:.2f}", delta=f"{change:.2f}")