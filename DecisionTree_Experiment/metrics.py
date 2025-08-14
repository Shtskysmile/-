import numpy as np
def accuracy_score(y_pred, y_true):
    return sum(y_true == y_pred) / len(y_true)

if __name__ == '__main__':
    y_pred = np.array([0, 0, 1])
    y_true = np.array([1, 1, 1])
    print(f"y_pred={y_pred}")
    print(f"y_true={y_true}")
    print(f"Accuracy Score: {accuracy_score(y_pred, y_true)}")