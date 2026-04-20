import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

class ArimaModel:
    def __init__(self):
        self.order = (5, 1, 0)
        self.model = None

    def find_best_order(self, prices: np.ndarray) -> tuple:
        best_aic = float("inf")
        best_order = self.order
        
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        
        for p in [1, 2, 3, 4, 5]:
            for d in [0, 1]:
                for q in [0, 1, 2]:
                    try:
                        model = ARIMA(prices, order=(p, d, q))
                        results = model.fit()
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_order = (p, d, q)
                    except:
                        continue
        self.order = best_order
        return best_order

    def train_and_predict(self, prices: np.ndarray, forecast_days: int = 30) -> dict:
        train_size = int(len(prices) * 0.8)
        train, test = prices[:train_size], prices[train_size:]
        
        self.find_best_order(train)
        
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        
        history = list(train)
        predictions = []
        
        # Limit rolling forecast for long test sets to avoid timeout
        test_limit = min(60, len(test))
        test_idx_start = len(test) - test_limit
        
        # If we have a reduced test set, let's just train on the full test prep
        # Actually standard walkforward requires full history.
        if test_idx_start > 0:
            history.extend(test[:test_idx_start])
            
        test_subset = test[test_idx_start:]
        for t in range(len(test_subset)):
            model = ARIMA(history, order=self.order)
            model_fit = model.fit()
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            history.append(test_subset[t])
            
        # Final model on full dataset for forecasting
        final_model = ARIMA(prices, order=self.order)
        final_fit = final_model.fit()
        forecast = final_fit.forecast(steps=forecast_days)
        
        mae = mean_absolute_error(test_subset, predictions)
        rmse = np.sqrt(mean_squared_error(test_subset, predictions))
        
        return {
            "test_actual": test_subset.tolist(),
            "test_predicted": [float(p) for p in predictions],
            "forecast": forecast.tolist(),
            "mae": float(mae),
            "rmse": float(rmse),
            "order": list(self.order)
        }
