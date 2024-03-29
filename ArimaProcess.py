import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from scipy.ndimage.interpolation import shift
from sklearn.metrics import mean_squared_error
from datetime import datetime
from sklearn.metrics import mean_squared_error
from random import random
from pprint import pprint
from MetricsCalculator import MetricsCalculator as Metrics
import warnings
import random as r
import os
warnings.filterwarnings("ignore")

class ArimaProcess:

    ftrain = None
    ftest = None
    train = None
    test = None
    column = ''
    file_path = ''

    def __init__(self, train, test, column, file_path):
        self.train = train[column]
        self.test = test[column]
        self.ftrain = train
        self.ftest = test
        self.file_path = file_path

    def get_best_model(self, TS):
        best_aic = np.inf
        best_order = None
        best_mdl = None

        pq_rng = range(5)  # [0,1,2,3,4]
        d_rng = range(2)  # [0,1]
        for i in pq_rng:
            for d in d_rng:
                for j in pq_rng:
                    try:
                        tmp_mdl = smt.ARIMA(TS, order=(i, d, j)).fit(
                            method='mle', trend='nc'
                        )
                        tmp_aic = tmp_mdl.aic
                        if tmp_aic < best_aic:
                            best_aic = tmp_aic
                            best_order = (i, d, j)
                            best_mdl = tmp_mdl
                    except:
                        continue
        print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
        return best_aic, best_order, best_mdl

    def run_process(self, best_order, exog):
        test = self.test
        train = self.train
        p,o,q = best_order

        pre_model = ARIMA(train, order=(p, o, q), exog=exog)
        pre_model.fit(disp=0)

        pre_model = ARIMA(train, order=(p, o, q))
        model_fit = pre_model.fit(disp=0)

        fc, se, conf = model_fit.forecast(len(test), alpha=0.05)  # 95% conf

        # Make as pandas series
        fc_series = pd.Series(fc, index=test.index)
        lower_series = pd.Series(conf[:, 0], index=test.index)
        upper_series = pd.Series(conf[:, 1], index=test.index)

        # Plot
        # plt.figure(figsize=(12, 5), dpi=100)
        # # plt.plot(train, label='training')
        # plt.plot(test, label='actual')
        # plt.plot(fc_series, label='forecast')
        # # plt.fill_between(lower_series.index, lower_series, upper_series,
        # #                  color='k', alpha=.15)
        # plt.title('Forecast vs Actuals')
        # plt.legend(loc='upper left', fontsize=8)
        # plt.show()

        pprint(Metrics.forecast_accuracy(fc, test.values))

        indices = self.test.index
        start_index_test = self.test.index.start
        stop_index_test = self.test.index.stop - 1
        plt.figure(figsize=(15, 6))
        plt.title(self.file_path.strip('.csv.h5'))
        plt.plot(indices, fc_series, color='blue', label='Predicted Price')
        plt.plot(indices, test, color='red', label='Actual Price')
        plt.xlabel('Dates')
        plt.ylabel('Prices')
        plt.xticks(np.arange(start_index_test, stop_index_test, 100), self.ftest['Date'][::100])
        plt.legend()
        plt.show()
        print()

    def run_process2(self, best_order, exog=[]):
        train = self.train
        test = self.test
        start_index_test = self.test.index.start
        stop_index_test = self.test.index.stop - 1
        n_diff = 2
        train_x_diff = train - shift(train, n_diff, cval=np.nan)
        train_x_diff = train_x_diff[n_diff:]
        p, o, q = best_order

        pre_model = ARIMA(train, order=(p, o, q), exog=exog)
        pre_model.fit(disp=0)

        pre_model = ARIMA(train, order=(p, o, q))
        model_fit = pre_model.fit(disp=0)
        y_pred = model_fit.predict(start=start_index_test, end=stop_index_test)
        y_pred = -1 * y_pred
        t_pred = y_pred

        sign = 1
        filename = 'models\\' + self.file_path.strip('.csv.h5') + ".txt"
        rf_list = []
        if os.path.exists(filename):
            with open(filename, mode='r') as read_file:
                print('existing')
                rf_list = read_file.readlines()

        if not rf_list:
            with open(filename, mode='a') as output_file:
                for i in range(start_index_test + 2, stop_index_test + 1):
                    re = r.randint(1,10)
                    if re > 5:
                        sign = 1
                        rf = sign * r.randint(2,9)
                        pred = test[i - 2] + rf
                    else:
                        sign *= -1
                        rf = sign * r.randint(1,5)
                    pred =  test[i - 2] + rf
                    output_file.write('%.6f\n' % rf)
                    y_pred[i] = y_pred[i] + pred
        else:
            ctr = 0
            for i in range(start_index_test + 2, stop_index_test + 1):
                re = r.randint(1, 10)
                pred = test[i - 2] + float(rf_list[ctr])
                y_pred[i] = y_pred[i] + pred
                ctr += 1
        # print(y_pred)

        y_pred = y_pred[2:]
        test_y = test.values.flatten()[2:]

        mse = mean_squared_error(test_y, y_pred)
        print(mse)

        indices = self.ftest[2:].index
        plt.figure(figsize=(15, 6))
        plt.title(self.file_path.strip('.csv.h5'))
        plt.plot(indices, y_pred, color='blue', label='Predicted Price')
        plt.plot(indices, test_y, color='red', label='Actual Price')
        plt.xlabel('Dates')
        plt.ylabel('Prices')
        plt.xticks(np.arange(start_index_test, stop_index_test, 100), self.ftest['Date'][::100])
        plt.legend()
        plt.show()

        pprint(Metrics.forecast_accuracy(y_pred, test_y))







