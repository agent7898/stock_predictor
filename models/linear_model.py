import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class LinearRegressionModel:

    def __init__(self):
        self.model = LinearRegression()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.look_back = 30

    def _make_sequences(self, arr):
        X, y = [], []
        for i in range(self.look_back, len(arr)):
            X.append(arr[i - self.look_back:i, 0])
            y.append(arr[i, 0])
        return np.array(X), np.array(y)

    def train_and_predict(self, prices: np.ndarray, forecast_days: int = 30) -> dict:
        prices = prices.reshape(-1, 1).astype(np.float64)

        # ── split raw prices 80/20 ──────────────────────────
        split = int(len(prices) * 0.8)
        train_raw = prices[:split]
        test_raw  = prices[split:]

        # ── scale: fit ONLY on train ────────────────────────
        train_scaled = self.scaler.fit_transform(train_raw)
        test_scaled  = self.scaler.transform(test_raw)

        # ── sequences ───────────────────────────────────────
        full_scaled = np.concatenate([train_scaled, test_scaled], axis=0)
        X, y = self._make_sequences(full_scaled)

        train_samples = len(train_scaled) - self.look_back
        X_train, y_train = X[:train_samples], y[:train_samples]
        X_test,  y_test  = X[train_samples:], y[train_samples:]

        # ── train ────────────────────────────────────────────
        self.model.fit(X_train, y_train)

        # ── predict on test (still scaled) ──────────────────
        y_pred_scaled = self.model.predict(X_test)

        # ── inverse transform to real prices ─────────────────
        actual_real = self.scaler.inverse_transform(
            y_test.reshape(-1, 1)).flatten()
        pred_real   = self.scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)).flatten()

        # ── metrics on REAL prices ────────────────────────────
        mae  = float(mean_absolute_error(actual_real, pred_real))
        rmse = float(np.sqrt(mean_squared_error(actual_real, pred_real)))
        r2   = float(r2_score(actual_real, pred_real))

        # ── iterative forecast ────────────────────────────────
        window = full_scaled[-self.look_back:].flatten().tolist()
        forecast_scaled = []
        for _ in range(forecast_days):
            x_in = np.array(window[-self.look_back:]).reshape(1, -1)
            p = float(self.model.predict(x_in)[0])
            forecast_scaled.append(p)
            window.append(p)

        forecast_real = self.scaler.inverse_transform(
            np.array(forecast_scaled).reshape(-1, 1)).flatten()

        return {
            "test_actual":    actual_real.tolist(),
            "test_predicted": pred_real.tolist(),
            "forecast":       forecast_real.tolist(),
            "mae":            mae,
            "rmse":           rmse,
            "r2":             r2
        }
