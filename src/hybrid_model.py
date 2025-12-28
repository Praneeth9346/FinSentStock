import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(LSTMNet, self).__init__()
        # Increased complexity slightly
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # Take the last time step output
        return self.fc(out[:, -1, :])

class HybridPredictor:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        # We need TWO scalers now: one for inputs (X), one for target (y)
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
        self.lstm_model = None
        self.lgb_model = None

    def prepare_data(self, df):
        """Prepares features (X) and target (y)"""
        # Features
        feature_cols = ['Close', 'RSI', 'MACD', 'EMA_20']
        if 'Sentiment' in df.columns:
            feature_cols.append('Sentiment')
            
        # 1. Prepare X (Features)
        X_raw = df[feature_cols].values
        X_raw = X_raw[:-1] # Drop last row (no target for it)
        
        # 2. Prepare y (Target)
        # We want to predict Next Day's Close
        y_raw = df['Close'].shift(-1).dropna().values.reshape(-1, 1)

        # 3. Scale Both
        X_scaled = self.scaler_X.fit_transform(X_raw)
        y_scaled = self.scaler_y.fit_transform(y_raw)
        
        # Flatten y for LightGBM compatibility
        return X_scaled, y_scaled.flatten()

    def train(self, X, y):
        # Split Data (80% Train, 20% Test)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # --- 1. Train LightGBM ---
        # LightGBM can handle raw data, but scaled is fine too.
        lgb_train = lgb.Dataset(X_train, y_train)
        params = {
            'objective': 'regression', 
            'metric': 'rmse', 
            'learning_rate': 0.05,
            'verbosity': -1
        }
        self.lgb_model = lgb.train(params, lgb_train, num_boost_round=500)

        # --- 2. Train LSTM ---
        # Reshape for LSTM: (Batch_Size, Sequence_Length, Features)
        # Here Sequence_Length = 1 (using current day to predict tomorrow)
        X_train_t = torch.FloatTensor(X_train).unsqueeze(1) 
        y_train_t = torch.FloatTensor(y_train).view(-1, 1)
        
        self.lstm_model = LSTMNet(input_dim=X.shape[1])
        criterion = nn.MSELoss()
        # Lower learning rate for stability
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.005)

        epochs = 100
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.lstm_model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()

        return self.evaluate(X_test, y_test)

    def evaluate(self, X_test, y_test):
        # 1. LightGBM Prediction (Scaled)
        lgb_pred_scaled = self.lgb_model.predict(X_test)
        
        # 2. LSTM Prediction (Scaled)
        X_test_t = torch.FloatTensor(X_test).unsqueeze(1)
        self.lstm_model.eval()
        with torch.no_grad():
            lstm_pred_scaled = self.lstm_model(X_test_t).numpy().flatten()
        
        # 3. Hybrid Ensemble (Scaled)
        final_pred_scaled = (self.alpha * lstm_pred_scaled) + ((1 - self.alpha) * lgb_pred_scaled)
        
        # 4. INVERSE SCALE (Convert back to Real Prices)
        final_pred_real = self.scaler_y.inverse_transform(final_pred_scaled.reshape(-1, 1)).flatten()
        y_test_real = self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Calculate RMSE on Real Prices
        rmse = np.sqrt(np.mean((y_test_real - final_pred_real)**2))
        return rmse, final_pred_real, y_test_real

    def predict_next(self, current_features):
        """Predicts tomorrow's price"""
        # Scale inputs
        scaled_feat = self.scaler_X.transform(current_features.reshape(1, -1))
        
        # LightGBM
        lgb_p = self.lgb_model.predict(scaled_feat)[0]
        
        # LSTM
        lstm_feat = torch.FloatTensor(scaled_feat).unsqueeze(1)
        self.lstm_model.eval()
        with torch.no_grad():
            lstm_p = self.lstm_model(lstm_feat).item()
        
        # Average
        final_p_scaled = (self.alpha * lstm_p) + ((1 - self.alpha) * lgb_p)
        
        # Inverse Scale to get Dollar Amount
        final_p_real = self.scaler_y.inverse_transform([[final_p_scaled]])[0][0]
        return final_p_real
