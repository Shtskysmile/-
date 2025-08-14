import numpy as np


class DataReader(object):
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

class DataProcessor(object):
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def split_label(self):
        X_train = self.train_data[:, 0:-1]
        Y_train = self.train_data[:, -1].astype(int)
        X_test = self.test_data[:, 0: -1]
        Y_test = self.test_data[:, -1].astype(int)
        return X_train, Y_train, X_test, Y_test

    def normalize(self, X_train, X_test):
        min_val = np.min(X_train, axis=0)
        max_val = np.max(X_train, axis=0)
        X_train_norm = (X_train - min_val) / (max_val - min_val)
        X_test_norm = (X_test - min_val) / (max_val - min_val)
        return X_train_norm, X_test_norm

class Dataset(object):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.dataset):
            raise StopIteration
        batch_indices = self.indices[\
            self.current_idx:self.current_idx + self.batch_size\
        ]
        batch = [self.dataset[i] for i in batch_indices]
        self.current_idx += self.batch_size
        batch_data, batch_labels = zip(*batch)
        return np.array(batch_data), np.array(batch_labels)