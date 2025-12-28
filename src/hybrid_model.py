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
        
        # 2. Validation
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # 3. Clean Data
        # FIX: 'Close' is already in feature_cols, so no need to add it again
        data_df = df[feature_cols].copy()
        
        # Handle Infinite and NaNs
        data_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        data_df.dropna(inplace=True)

        if len(data_df) < 10:
             st.error(f"Not enough data points. Need at least 10, found {len(data_df)}.")
             st.stop()

        # 4. Shift Data (Target is Next Day's Close)
        # X is all rows except the last one
        X_raw = data_df[feature_cols].values[:-1]
        
        # y is Close price shifted by -1 (the next day)
        # Since 'Close' is in feature_cols, we can access it directly
        y_raw = data_df['Close'].values[1:]

        # Reshape for Scaler
        y_raw = y_raw.reshape(-1, 1)

        # 5. Scale
        X_scaled = self.scaler_X.fit_transform(X_raw)
        y_scaled = self.scaler_y.fit_transform(y_raw)
        
        return X_scaled, y_scaled.flatten()

    def train(self, X, y):
        if len(X) < 5:
            return 0.0, np.array([]), np.array([])

        split = int(len(X) * 0.8)
        if split >= len(X):
            split = len(X) - 1

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # --- Train LightGBM ---
        lgb_train = lgb.Dataset(X_train, y_train)
        params = {
            'objective': 'regression', 
            'metric': 'rmse', 
            'learning_rate': 0.05, 
            'verbosity': -1,
            'min_data_in_leaf': 2
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
        # 1. LightGBM Prediction
        lgb_pred_scaled = self.lgb_model.predict(X_test).flatten()
        
        # 2. LSTM Prediction
        X_test_t = torch.FloatTensor(X_test).unsqueeze(1)
        self.lstm_model.eval()
        with torch.no_grad():
            lstm_pred_scaled = self.lstm_model(X_test_t).detach().numpy().flatten()
        
        # 3. Ensemble
        min_len = min(len(lgb_pred_scaled), len(lstm_pred_scaled))
        lgb_pred_scaled = lgb_pred_scaled[:min_len]
        lstm_pred_scaled = lstm_pred_scaled[:min_len]
        
        final_pred_scaled = (self.alpha * lstm_pred_scaled) + ((1 - self.alpha) * lgb_pred_scaled)
        
        # 4. Inverse Scale
        final_pred_real = self.scaler_y.inverse_transform(final_pred_scaled.reshape(-1, 1)).flatten()
        y_test_real = self.scaler_y.inverse_transform(y_test[:min_len].reshape(-1, 1)).flatten()
        
        rmse = np.sqrt(np.mean((y_test_real - final_pred_real)**2))
        return rmse, final_pred_real, y_test_real

    def predict_next(self, current_features):
        scaled_feat = self.scaler_X.transform(current_features.reshape(1, -1))
        
        lgb_p = self.lgb_model.predict(scaled_feat)[0]
        
        lstm_feat = torch.FloatTensor(scaled_feat).unsqueeze(1)
        self.lstm_model.eval()
        with torch.no_grad():
            lstm_p = self.lstm_model(lstm_feat).detach().item()
        
        final_p_scaled = (self.alpha * lstm_p) + ((1 - self.alpha) * lgb_p)
        final_p_real = self.scaler_y.inverse_transform([[final_p_scaled]])[0][0]
        return final_p_real
