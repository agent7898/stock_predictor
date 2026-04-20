import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error

class LSTMModel:
    def __init__(self):
        self.look_back = 60
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def build_model(self, input_shape: tuple):
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def create_sequences(self, data: np.ndarray):
        X, y = [], []
        for i in range(self.look_back, len(data)):
            X.append(data[i-self.look_back:i, 0])
            y.append(data[i, 0])
        return np.array(X).reshape(-1, self.look_back, 1), np.array(y)

    def train_and_predict(self, prices: np.ndarray, forecast_days: int = 30) -> dict:
        prices_scaled = self.scaler.fit_transform(prices.reshape(-1, 1))
        X, y = self.create_sequences(prices_scaled)
        
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        self.build_model((self.look_back, 1))
        self.model.fit(X_train, y_train, batch_size=32, epochs=20, 
                       validation_data=(X_test, y_test), verbose=0)
                       
        y_pred = self.model.predict(X_test)
        
        # FIX: Retrain the model on the FULL 100% dataset so it learns the most recent 20% of prices
        self.model.fit(X, y, batch_size=32, epochs=10, verbose=0)
        
        # Forecast
        forecast = []
        current_window = prices_scaled[-self.look_back:]
        
        for _ in range(forecast_days):
            pred = self.model.predict(current_window.reshape((1, self.look_back, 1)), verbose=0)
            forecast.append(pred[0, 0])
            current_window = np.append(current_window[1:], pred[0, 0])
            
        test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        test_predicted = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        forecast_inv = self.scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
        
        mae = mean_absolute_error(test_actual, test_predicted)
        rmse = np.sqrt(mean_squared_error(test_actual, test_predicted))
        
        return {
            "test_actual": test_actual.tolist(),
            "test_predicted": test_predicted.tolist(),
            "forecast": forecast_inv.tolist(),
            "mae": float(mae),
            "rmse": float(rmse)
        }
