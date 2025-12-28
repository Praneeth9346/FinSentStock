import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=50):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class HybridPredictor:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.scaler = MinMaxScaler()
        self.lstm_model = None
        self.lgb_model = None

    def prepare_data(self, df):
        """Prepares features (X) and target (y)"""
        # Features: Close, RSI, MACD, EMA, Sentiment (if available)
        feature_cols = ['Close', 'RSI', 'MACD', 'EMA_20']
        if 'Sentiment' in df.columns:
            feature_cols.append('Sentiment')
            
        data = df[feature_cols].values
        target = df['Close'].shift(-1).dropna().values # Predict Next Day Close
        data = data[:-1] # Drop last row as it has no target
        
        return self.scaler.fit_transform(data), target

    def train(self, X, y):
        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # 1. Train LightGBM
        lgb_train = lgb.Dataset(X_train, y_train)
        params = {'objective': 'regression', 'metric': 'rmse', 'verbosity': -1}
        self.lgb_model = lgb.train(params, lgb_train, num_boost_round=100)

        # 2. Train LSTM
        X_train_t = torch.FloatTensor(X_train).unsqueeze(1) # (Batch, Seq, Feature)
        y_train_t = torch.FloatTensor(y_train).view(-1, 1)
        
        self.lstm_model = LSTMNet(input_dim=X.shape[1])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.01)

        # Simple Training Loop
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = self.lstm_model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()

        return self.evaluate(X_test, y_test)

    def evaluate(self, X_test, y_test):
        # LightGBM Pred
        lgb_pred = self.lgb_model.predict(X_test)
        
        # LSTM Pred
        X_test_t = torch.FloatTensor(X_test).unsqueeze(1)
        lstm_pred = self.lstm_model(X_test_t).detach().numpy().flatten()
        
        # Ensemble
        final_pred = (self.alpha * lstm_pred) + ((1 - self.alpha) * lgb_pred)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((y_test - final_pred)**2))
        return rmse, final_pred, y_test

    def predict_next(self, current_features):
        """Predicts tomorrow's price"""
        scaled_feat = self.scaler.transform(current_features.reshape(1, -1))
        
        # LightGBM
        lgb_p = self.lgb_model.predict(scaled_feat)[0]
        
        # LSTM
        lstm_feat = torch.FloatTensor(scaled_feat).unsqueeze(1)
        lstm_p = self.lstm_model(lstm_feat).detach().item()
        
        final_p = (self.alpha * lstm_p) + ((1 - self.alpha) * lgb_p)
        return final_p