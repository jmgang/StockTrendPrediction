import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

class PlotMaker:

    @staticmethod
    def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        with plt.style.context(style):
            fig = plt.figure(figsize=figsize)
            # mpl.rcParams['font.family'] = 'Ubuntu Mono'
            layout = (3, 2)
            ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
            acf_ax = plt.subplot2grid(layout, (1, 0))
            pacf_ax = plt.subplot2grid(layout, (1, 1))
            qq_ax = plt.subplot2grid(layout, (2, 0))
            pp_ax = plt.subplot2grid(layout, (2, 1))

            y.plot(ax=ts_ax)
            ts_ax.set_title('Time Series Analysis Plots')
            smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
            smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
            sm.qqplot(y, line='s', ax=qq_ax)
            qq_ax.set_title('QQ Plot')
            scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

            plt.tight_layout()
            plt.show()
        return

    @staticmethod
    def simple_plot(TS):
        plt.plot(TS)
        plt.show()

    @staticmethod
    def standard_plot(title, actual, predicted, actual_label, predicted_label, xlabel, ylabel, legend=True):
        plt.plot(actual, color='red', label=actual_label)
        plt.plot(predicted, color='blue', label=predicted_label)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend() if legend else None
        plt.show()

    @staticmethod
    def timeseries_plot(indices, actual, predicted, actual_label, predicted_label, xlabel, ylabel, start_index, end_index, actual_index, plot_title):
        plt.figure(figsize=(15, 6))
        plt.title(plot_title)
        plt.plot(indices, predicted, color='blue', label=predicted_label)
        plt.plot(indices, actual, color='red', label=actual_label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(np.arange(start_index, end_index, 100), actual_index)
        plt.legend()
        plt.show()