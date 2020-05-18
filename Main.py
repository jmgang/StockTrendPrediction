
from ProcessingManager import ProcessingManager

if __name__ == "__main__":
    directory = 'data/'
    manager = ProcessingManager(directory, algorithm='LSTM')
    manager.process()