# 📈 FinSentStock: News Sentiment to Predict Short-Term Stock Movement

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20LightGBM-orange)
![NLP](https://img.shields.io/badge/NLP-FinBERT-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

**FinSentStock** is a hybrid financial forecasting system that integrates **Natural Language Processing (NLP)** with **Machine Learning** to predict short-term stock price movements. Unlike traditional models that rely solely on historical price data, FinSentStock fuses quantitative technical indicators with qualitative sentiment analysis derived from financial news.

## 🚀 Features

* **Hybrid Architecture:** Combines **LSTM** (for temporal pattern recognition) and **LightGBM** (for feature-based optimization) in a weighted ensemble.
* **Sentiment Analysis:** Utilizes **FinBERT** (a BERT model fine-tuned on financial text) to extract sentiment polarity (Positive, Negative, Neutral) from news headlines.
* **Technical Analysis:** Automatically computes key indicators including **RSI** (Relative Strength Index), **MACD** (Moving Average Convergence Divergence), and **EMA** (Exponential Moving Average).
* **Ensemble Prediction:** Final prediction is calculated using a weighted average:
    $$P_{final} = \alpha \times P_{LSTM} + (1-\alpha) \times P_{LightGBM}$$

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Deep Learning:** PyTorch (LSTM), Hugging Face Transformers (FinBERT)
* **Machine Learning:** LightGBM, Scikit-learn
* **Data Sources:** Yahoo Finance API (`yfinance`), NewsAPI
* **Visualization:** Matplotlib, Seaborn

## 📊 System Architecture

The system operates in a multi-stage pipeline:
1.  **Data Collection:** Fetches OHLCV data and financial news.
2.  **Sentiment Extraction:** News headlines are processed via FinBERT to generate daily sentiment scores.
3.  **Feature Engineering:** Merges market data, technical indicators, and sentiment scores.
4.  **Hybrid Training:** The LSTM and LightGBM branches are trained independently and fused for the final output.

## 💻 Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/](https://github.com/)[YourUsername]/FinSentStock.git
    cd FinSentStock
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set API Keys**
    Create a `.env` file and add your NewsAPI key:
    ```
    NEWS_API_KEY=your_api_key_here
    ```

## 🧠 Usage

You can run the full pipeline using the main script:

```python
from src.hybrid_model import FinSentStock

# Initialize the model
model = FinSentStock(ticker="AAPL", start_date="2023-01-01")

# 1. Fetch Data & Extract Sentiment
data = model.prepare_data()

# 2. Train Hybrid Model
model.train()

# 3. Predict Next Day Movement
prediction = model.predict_next_day()
print(f"Prediction for tomorrow: {prediction}")


📄 Documentation
This project is based on the Major Project "FinSentStock: News Sentiment to Predict Short-Term Stock Movement" submitted to B. V. Raju Institute of Technology.

Key References:

FinBERT: Financial Sentiment Analysis with Pre-trained Language Models (Araci, 2019)

LightGBM: A Highly Efficient Gradient Boosting Decision Tree (Ke et al., 2017)

🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.

📝 License
Distributed under the MIT License.