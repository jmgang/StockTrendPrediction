# Importing the Keras libraries and packages
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from PlotMaker import PlotMaker
from MetricsCalculator import MetricsCalculator
from pprint import pprint
import os
import warnings
warnings.filterwarnings("ignore")

class LSTM_Model:

    dataset = None
    train = None
    test = None
    ftrain = None
    ftest = None
    regressor = Sequential()
    X_train, y_train = None, None
    sc = MinMaxScaler(feature_range = (0, 1))
    file_path = None
    does_model_exist = False
    directory = 'models'


    def __init__(self, dataset, train, test, column, file_path, directory):
        self.dataset = dataset
        self.train = train[column]
        self.test = test[column]
        self.ftrain = train
        self.ftest = test
        self.file_path = file_path
        self.directory = directory
        self.is_there_an_existing_model(directory_str=self.directory)

    def normalize_dataset(self):
        self.train = self.train.values.reshape(-1,1)
        # print(self.train)
        training_set_scaled = self.sc.fit_transform(self.train)

        X_train = []
        y_train = []
        for i in range(60, len(training_set_scaled)):
            X_train.append(training_set_scaled[i - 60:i, 0])
            y_train.append(training_set_scaled[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)

        # Reshaping
        self.X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        self.y_train = y_train

    def create_lstm_model(self):
        self.normalize_dataset()
        if self.does_model_exist:
            file_path = self.directory + self.file_path + '.h5'
            self.regressor = load_model(file_path)
            return
        # Adding the first LSTM layer and some Dropout regularisation
        self.regressor.add(LSTM(units=50, return_sequences=True, input_shape=(self.X_train.shape[1], 1)))
        self.regressor.add(Dropout(0.2))

        # Adding a second LSTM layer and some Dropout regularisation
        self.regressor.add(LSTM(units=50, return_sequences=True))
        self.regressor.add(Dropout(0.2))

        # Adding a third LSTM layer and some Dropout regularisation
        self.regressor.add(LSTM(units=50, return_sequences=True))
        self.regressor.add(Dropout(0.2))

        # Adding a fourth LSTM layer and some Dropout regularisation
        self.regressor.add(LSTM(units=50))
        self.regressor.add(Dropout(0.2))

        # Adding the output layer
        self.regressor.add(Dense(units=1))


    def train_lstm(self, epoch=1):
        if self.does_model_exist:
            return

        # Compiling the RNN
        self.regressor.compile(optimizer='adam', loss='mean_squared_error')

        # Fitting the RNN to the Training set
        self.regressor.fit(self.X_train, self.y_train, epochs=epoch, batch_size=32)

        file_path = self.directory + self.file_path + '.h5'
        self.regressor.save(file_path)
        print('model saved as ' + file_path)

    def predict(self):
        dataset = self.dataset
        test = self.test
        inputs = dataset[len(dataset) - len(test) - 60:].values
        inputs = inputs.reshape(-1, 1)
        inputs = self.sc.transform(inputs)
        X_test = []
        range_test_max = 60 + len(test)
        for i in range(60, range_test_max):
            X_test.append(inputs[i - 60:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_stock_price = self.regressor.predict(X_test)
        predicted_stock_price = self.sc.inverse_transform(predicted_stock_price)
        return predicted_stock_price

    def plot(self, predicted):
        title = self.file_path.strip('.csv.h5')
        test = self.test
        indices = test.index
        start_index_test = test.index.start
        stop_index_test = test.index.stop - 1
        dates_index = self.ftest['Date'][::100]
        PlotMaker.timeseries_plot(indices, test, predicted, 'Actual', 'Predicted', 'Dates', 'Prices', start_index_test, stop_index_test, dates_index, title)

        pprint(MetricsCalculator.forecast_accuracy(predicted.flatten(), test))


    def is_there_an_existing_model(self, directory_str):
        directory = os.fsencode(directory_str)
        for file in os.listdir(directory):
            file_path = os.fsdecode(file)
            if file_path == self.file_path + '.h5':
                self.does_model_exist = True