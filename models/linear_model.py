import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = MinMaxScaler()
        self.look_back = 30
        
    def prepare_features(self, prices: np.ndarray):
        X, y = [], []
        for i in range(self.look_back, len(prices)):
            X.append(prices[i-self.look_back:i])
            y.append(prices[i])
        return np.array(X), np.array(y)
        
    def train_and_predict(self, prices: np.ndarray, forecast_days: int = 30) -> dict:
        prices_scaled = self.scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        X, y = self.prepare_features(prices_scaled)
        
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        # FIX: Retrain on the FULL dataset so its final mathematical trajectory starts from the exact latest data
        self.model.fit(X, y)
        
        # Forecast future values
        forecast = []
        current_window = prices_scaled[-self.look_back:].copy()
        
        for _ in range(forecast_days):
            pred = self.model.predict([current_window])[0]
            forecast.append(pred)
            current_window = np.append(current_window[1:], pred)
            
        test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        test_predicted = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        forecast_inv = self.scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
        
        mae = mean_absolute_error(test_actual, test_predicted)
        rmse = np.sqrt(mean_squared_error(test_actual, test_predicted))
        r2 = r2_score(test_actual, test_predicted)
        
        return {
            "test_actual": test_actual.tolist(),
            "test_predicted": test_predicted.tolist(),
            "forecast": forecast_inv.tolist(),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2)
        }
