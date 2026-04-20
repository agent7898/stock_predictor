import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '42'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import random
random.seed(42)

import numpy as np
np.random.seed(42)

import tensorflow as tf
tf.random.set_seed(42)

import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping


class LSTMModel:

    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.seq_len = 60

    def _create_sequences(self, data: np.ndarray) -> tuple:
        x, y = [], []
        for i in range(self.seq_len, len(data)):
            x.append(data[i - self.seq_len: i, 0])
            y.append(data[i, 0])
        return np.array(x), np.array(y)

    def train_and_predict(self, prices: np.ndarray, forecast_days: int = 30) -> dict:
        random.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)
        
        prices = prices.astype(np.float64).reshape(-1, 1)

        split = int(len(prices) * 0.8)
        train_raw = prices[:split]
        test_raw  = prices[split:]

        if len(train_raw) <= self.seq_len:
            return {}

        # ── SCALING ONLY ON TRAIN ─────────────────────────────
        self.scaler.fit(train_raw)

        # Create properly scaled datasets
        train_scaled = self.scaler.transform(train_raw)

        # For test sequences, we need the last `seq_len` days from train
        # to predict the first day of test
        model_inputs_raw = prices[len(prices) - len(test_raw) - self.seq_len:]
        model_inputs_scaled = self.scaler.transform(model_inputs_raw)

        # ── SEQUENCES ─────────────────────────────────────────
        x_train, y_train = self._create_sequences(train_scaled)
        x_test, y_test   = self._create_sequences(model_inputs_scaled)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test  = np.reshape(x_test,  (x_test.shape[0],  x_test.shape[1],  1))

        # ── BUILD & TRAIN MODEL ───────────────────────────────
        tf.random.set_seed(42)
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(x_train, y_train, batch_size=32, epochs=20, callbacks=[es], verbose=0)

        # ── PREDICT TEST DATA ─────────────────────────────────
        preds_scaled = model.predict(x_test, verbose=0)

        # ── INVERSE TRANSFORM & METRICS ───────────────────────
        # Predictions shape back to real prices
        preds_real = self.scaler.inverse_transform(preds_scaled).flatten()
        actual_real = self.scaler.inverse_transform(
            y_test.reshape(-1, 1)).flatten()

        mae  = float(mean_absolute_error(actual_real, preds_real))
        rmse = float(np.sqrt(mean_squared_error(actual_real, preds_real)))

        # ── FORECAST FUTURE ───────────────────────────────────
        full_scaled = self.scaler.transform(prices)
        window = full_scaled[-self.seq_len:].reshape(self.seq_len, 1).copy()
        forecast_scaled = []
        last_real_price = float(prices[-1][0])
        lower_real = last_real_price * 0.85
        upper_real = last_real_price * 1.15
        prev_real = last_real_price

        for _ in range(forecast_days):
            x_in = window.reshape(1, self.seq_len, 1)
            raw_scaled = float(model.predict(x_in, verbose=0)[0][0])

            # Clamp forecast in real-price space to avoid compounding drift.
            raw_real = float(self.scaler.inverse_transform(
                np.array([[raw_scaled]])
            )[0][0])

            # Keep each step smooth and also bounded to a realistic global range.
            step_low = prev_real * 0.985
            step_high = prev_real * 1.015
            low_bound = max(lower_real, step_low)
            high_bound = min(upper_real, step_high)
            clamped_real = float(np.clip(raw_real, low_bound, high_bound))
            clamped_scaled = float(self.scaler.transform(
                np.array([[clamped_real]])
            )[0][0])

            forecast_scaled.append(clamped_scaled)
            window = np.vstack([window[1:], [[clamped_scaled]]])
            prev_real = clamped_real

        forecast_real = self.scaler.inverse_transform(
            np.array(forecast_scaled).reshape(-1, 1)).flatten()

        return {
            "test_actual":    [float(v) for v in actual_real],
            "test_predicted": [float(v) for v in preds_real],
            "forecast":       [float(v) for v in forecast_real],
            "mae":            mae,
            "rmse":           rmse
        }
