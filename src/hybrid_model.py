import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class HybridPredictor:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.lstm_model = None
        self.lgb_model = None

    def prepare_data(self, df):
        """Prepares features (X) and target (y) with robust cleaning"""
        # 1. Define Features
        feature_cols = ['Close', 'RSI', 'MACD', 'EMA_20']
        if 'Sentiment' in df.columns:
            feature_cols.append('Sentiment')
        
        # 2. Safety Check: Ensure columns exist
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")

        # 3. Clean Data (Handle Infinite and NaNs)
        # Create a copy to avoid SettingWithCopy warnings
        data_df = df[feature_cols + ['Close']].copy() 
        
        # Replace Infinity with NaN and drop all NaNs
        data_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        data_df.dropna(inplace=True)

        if len(data_df) < 10:
             st.error(f"Not enough data points after cleaning. Found {len(data_df)} rows. Need at least 10.")
             st.stop()

        # 4. separate X (Features) and y (Target)
        # We use the current day's features to predict the NEXT day's Close
        # So X is rows [0 to N-1], y is Close [1 to N]
        
        X_raw = data_df[feature_cols].values[:-1]  # All rows except last
        y_raw = data_df['Close'].values[1:]        # All rows except first (shifted -1)

        # Reshape y for scaler
        y_raw = y_raw.reshape(-1, 1)

        # 5. Scale
        X_scaled = self.scaler_X.fit_transform(X_raw)
        y_scaled = self.scaler_y.fit_transform(y_raw)
        
        return X_scaled, y_scaled.flatten()

    def train(self, X, y):
        # Basic split
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # --- Train LightGBM ---
        lgb_train = lgb.Dataset(X_train, y_train)
        params = {
            'objective': 'regression', 
            'metric': 'rmse', 
            'learning_rate': 0.05,
            'verbosity': -1,
            'min_data_in_leaf': 2  # Added to prevent errors on small datasets
        }
        self.lgb_model = lgb.train(params, lgb_train, num_boost_round=500)

        # --- Train LSTM ---
        X_train_t = torch.FloatTensor(X_train).unsqueeze(1) 
        y_train_t = torch.FloatTensor(y_train).view(-1, 1)
        
        self.lstm_model = LSTMNet(input_dim=X.shape[1])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.005)

        for epoch in range(100):
            optimizer.zero_grad()
            outputs = self.lstm_model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()

        return self.evaluate(X_test, y_test)

    def evaluate(self, X_test, y_test):
        # LightGBM
        lgb_pred_scaled = self.lgb_model.predict(X_test)
        
        # LSTM
        X_test_t = torch.FloatTensor(X_test).unsqueeze(1)
        self.lstm_model.eval()
        with torch.no_grad():
            lstm_pred_scaled = self.lstm_model(X_test_t).numpy().flatten()
        
        # Ensemble
        final_pred_scaled = (self.alpha * lstm_pred_scaled) + ((1 - self.alpha) * lgb_pred_scaled)
        
        # Inverse Scale
        final_pred_real = self.scaler_y.inverse_transform(final_pred_scaled.reshape(-1, 1)).flatten()
        y_test_real = self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        rmse = np.sqrt(np.mean((y_test_real - final_pred_real)**2))
        return rmse, final_pred_real, y_test_real

    def predict_next(self, current_features):
        scaled_feat = self.scaler_X.transform(current_features.reshape(1, -1))
        
        lgb_p = self.lgb_model.predict(scaled_feat)[0]
        
        lstm_feat = torch.FloatTensor(scaled_feat).unsqueeze(1)
        self.lstm_model.eval()
        with torch.no_grad():
            lstm_p = self.lstm_model(lstm_feat).item()
        
        final_p_scaled = (self.alpha * lstm_p) + ((1 - self.alpha) * lgb_p)
        final_p_real = self.scaler_y.inverse_transform([[final_p_scaled]])[0][0]
        return final_p_real
