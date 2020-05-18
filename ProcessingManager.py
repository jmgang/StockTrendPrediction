
import os
from os import path
import yaml
import ast
import pandas as pd
from ArimaProcess import ArimaProcess
from GARCH_Process import GARCH_Process
from LSTM import LSTM_Model

class ProcessingManager:

    directory_str = ''
    column = ''
    train_x = None
    test_y = None
    test_size = 0.8
    yaml_file_path = r'models/best_model_list.yaml'
    algorithm = ''


    def __init__(self, directory_str='', column='Adj Close', test_size=0.8, algorithm='LSTM'):
        self.directory_str = directory_str
        self.column = column
        self.test_size = test_size
        self.algorithm = algorithm

    def process(self):
        directory = os.fsencode(self.directory_str)
        for file in os.listdir(directory):
            file_path = os.fsdecode(file)
            filename, file_ext = os.path.splitext(file_path)
            full_filepath = self.directory_str + file_path
            df = pd.read_csv(full_filepath)
            train, test = self.train_test_split(df)

            if self.algorithm == 'LSTM':
                self.LSTM(df[self.column], train, test, file_path)
            elif self.algorithm == 'GARCH':
                self.ARIMA_GARCH(train, test, file_path)
            elif self.algorithm == 'BOTH':
                self.LSTM(df[self.column], train, test, file_path)
                self.ARIMA_GARCH(train, test, file_path)
            else:
                print('Please input correct algorithm type. Valid values are \'LSTM\', \'GARCH\' or \'BOTH\' ')

    def LSTM(self, dataset, train, test, file_path):
        print('processing company {0}\n'.format(file_path))
        lstm_model = LSTM_Model(dataset, train, test, self.column, file_path, directory='models/')
        lstm_model.create_lstm_model()
        lstm_model.train_lstm(epoch=100)
        predicted = lstm_model.predict()
        lstm_model.plot(predicted)


    def ARIMA_GARCH(self, train, test, file_path):
        company = file_path.strip('.csv')
        print('processing company {0}...'.format(company))
        arima_process = ArimaProcess(train, test, self.column, file_path)
        train, test = train[self.column], test[self.column]
        # check if yaml file is existing
        # if existing
        if path.exists(self.yaml_file_path):
            with open(self.yaml_file_path) as yaml_file:
                best_model_list = yaml.load(yaml_file, Loader=yaml.FullLoader)
            if company in best_model_list.keys():
                best_order = ast.literal_eval(best_model_list[company]['best_order'])
                print('best order: {0}'.format(best_order))
            else:
                best_aic, best_order, best_mdl = arima_process.get_best_model(train)
                best_model_list[company] = {'aic' : best_aic, 'best_order' : str(best_order)}
                self.dump_yaml(best_model_list)
        else:
            print('path not existing...\ncreating new yaml file...')
            # if not existing, run get_best_model, then create yaml file with the best model
            best_aic, best_order, best_mdl = arima_process.get_best_model(train)
            best_model_list = {company: {'aic': best_aic, 'best_order': str(best_order)}}
            self.dump_yaml(best_model_list)

        garch_process = GARCH_Process( best_order[0], best_order[1], best_order[2], train, test )
        garch_process.process_garch(arima_process)

    def train_test_split(self, df):
        return df[0:int(len(df) * self.test_size)], df[int(len(df) * self.test_size):]

    def is_company_existing(self, company, yaml_list):
        pass

    def dump_yaml(self, model_list):
        with open(self.yaml_file_path, 'w') as yaml_output:
            documents = yaml.dump(model_list, yaml_output)
            print('successfully dumped yaml file')