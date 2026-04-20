import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


class ArimaModel:

    def __init__(self):
        self.order = (5, 1, 0)

    def _find_best_order(self, train: np.ndarray) -> tuple:
        best_aic   = np.inf
        best_order = (1, 1, 0)
        for p in range(1, 6):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        aic = ARIMA(train, order=(p, d, q)).fit().aic
                        if aic < best_aic:
                            best_aic   = aic
                            best_order = (p, d, q)
                    except Exception:
                        continue
        return best_order

    def train_and_predict(self, prices: np.ndarray, forecast_days: int = 30) -> dict:
        # ARIMA works on raw prices — NO scaling at all
        prices = prices.astype(np.float64)

        split = int(len(prices) * 0.8)
        train = prices[:split]
        test  = prices[split:]

        # ── find best order on training data only ─────────────
        self.order = self._find_best_order(train)

        # ── rolling forecast on full test window ─────────────
        cap     = len(test)
        history = list(train)
        predictions = []

        for i in range(cap):
            try:
                fitted = ARIMA(history, order=self.order).fit()
                pred   = float(fitted.forecast(steps=1)[0])
            except Exception:
                pred = float(history[-1])
            predictions.append(pred)
            history.append(float(test[i]))

        test_actual    = [float(v) for v in test[:cap]]
        test_predicted = predictions

        # ── metrics on REAL prices ────────────────────────────
        mae  = float(mean_absolute_error(test_actual, test_predicted))
        rmse = float(np.sqrt(mean_squared_error(test_actual, test_predicted)))

        # ── final forecast from full dataset ──────────────────
        try:
            final_model  = ARIMA(prices, order=self.order).fit()
            forecast_raw = final_model.forecast(steps=forecast_days)
            forecast     = [float(v) for v in forecast_raw]
        except Exception:
            last = float(prices[-1])
            forecast = [last] * forecast_days

        return {
            "test_actual":    test_actual,
            "test_predicted": test_predicted,
            "forecast":       forecast,
            "mae":            mae,
            "rmse":           rmse,
            "order":          [int(self.order[0]),
                               int(self.order[1]),
                               int(self.order[2])]
        }
