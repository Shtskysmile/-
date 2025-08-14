import argparse
import numpy as np
from model import MLP
from utils import train_one_epoch
from metric import accuracy_score
from dataset import DataReader, Dataset, DataLoader, DataProcessor


def main(args):
    # np.random.seed(args.seed)
    data_reader = DataReader(
        train_path='./data/Iris-train.txt',
        test_path='./data/Iris-test.txt'
    )
    train_data, test_data = data_reader.load_data()
    data_processor = DataProcessor(train_data, test_data)
    X_train, y_train, X_test, y_test = data_processor.split_label()
    X_train, X_test = data_processor.normalize(X_train, X_test)
    train_dataset = Dataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    accuracy_list = []
    for s in range(10):
        np.random.seed(s)
        model = MLP(args.input_size, args.hidden_size, args.output_size)

        for epoch in range(args.epochs):
            for batch_X, batch_y in train_loader:
                loss = train_one_epoch(model, batch_X, batch_y, learning_rate=args.learning_rate)
            # print(f'Epoch {epoch + 1}/{args.epochs}, Loss: {loss:.4f}')

        # Evaluate the model
        y_pred = np.argmax(model.forward(X_test), axis=1)
        accuracy = accuracy_score(y_pred, y_test)
        accuracy_list.append(accuracy)
        print(f'Seed {s} Accuracy: {accuracy:.4f}')

    print(f"Average Accuracy: {np.array(accuracy_list).mean():4f}")
    print(f"Accuracy Std: {np.array(accuracy_list).std():4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an MLP model.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--input_size', type=int, default=4, help='Input size of the model')
    parser.add_argument('--hidden_size', type=int, default=10, help='Hidden layer size of the model')
    parser.add_argument('--output_size', type=int, default=3, help='Output size of the model')
    parser.add_argument('--seed', type=int, default=10, help='Seed for np.random')
    args = parser.parse_args()
    main(args)
