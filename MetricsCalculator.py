
from sklearn import metrics
import numpy as np

class MetricsCalculator:

    @staticmethod
    def forecast_accuracy(forecast, actual):
        mape = np.mean(np.abs(forecast - actual) / np.abs(actual))  # MAPE
        me = np.mean(forecast - actual)  # ME
        mae = np.mean(np.abs(forecast - actual))  # MAE
        mpe = np.mean((forecast - actual) / actual)  # MPE
        rmse = np.mean((forecast - actual) ** 2) ** .5  # RMSE
        accuracy = 100 - ( mape * 100 )
        return ({'mape': mape, 'me': me, 'mae': mae,
                 'mpe': mpe, 'rmse': rmse, 'accuracy' : accuracy})
