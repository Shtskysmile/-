from DecisionTree import DecisionTreeClassifier
from data_loader import DataLoader
import argparse
from metrics import accuracy_score

def main(args):
    max_depth = args.max_depth
    model = DecisionTreeClassifier(max_depth=max_depth)
    data_loader = DataLoader('data/traindata.txt', 'data/testdata.txt')
    train_data, test_data = data_loader.load_data()
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: {:.2f}'.format(accuracy))
    # model.print_tree()
    model.plot_tree()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_depth', type=int, default=5)
    args = parser.parse_args()
    main(args)
