
from arch import arch_model
from PlotMaker import PlotMaker
import numpy as np

class GARCH_Process:

    p, o, q = 0, 0, 0
    time_series = 0
    n_test = 0
    train = None
    test = None

    def __init__(self, p, o, q, time_series, test):
        self.p, self.o, self.q = p, o, q
        self.time_series = time_series
        self.test = test
        self.n_test = len(time_series)

    def process_garch(self, ap):
        am = arch_model(self.time_series, p=self.p, o=self.o, q=self.q, vol='GARCH', dist='StudentsT')
        res = am.fit(update_freq=5, disp='off')

        print('forecasting...\nplease wait...')
        res_hat = res.forecast(horizon=self.n_test)
        res_val = res_hat.residual_variance.values[-1,:-1]
        res_val = np.append(res_val, [0])
        # ap.run_process([self.p, self.o, self.q], exog=res_val)
        ap.run_process2([self.p, self.o, self.q], exog=res_val)






