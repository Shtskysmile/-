import numpy as np

class DataLoader(object):
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

    def load_data(self):
        train_data = []
        test_data = []
        with open(self.train_path, 'r') as f:
            for line in f:
                train_data.append(list(map(float, line.split())))

        with open(self.test_path, 'r') as f:
            for line in f:
                test_data.append(list(map(float, line.split())))

        return np.array(train_data), np.array(test_data)
    
if __name__ == '__main__':
    dataloader = DataLoader('data/traindata.txt', 'data/testdata.txt')
    print(dataloader.load_data())