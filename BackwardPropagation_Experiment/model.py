import numpy as np

class MLP(object):
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def softmax(self, Z):
        expZ = np.exp(Z)
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = np.dot(X, self.w1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.w2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def calculate_loss(self, X, y):
        m = y.shape[0]
        A2 = self.forward(X)
        log_probs = -np.log(A2[range(m), y])
        loss = np.sum(log_probs) / m    # 一个批量平均
        return loss

    def backprop(self, X, y, learning_rate=0.01):
        m = y.shape[0]
        A2 = self.A2
        A2[range(m), y] -= 1
        dZ2 = A2 / m

        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)    # batch维度求和
        dA1 = np.dot(dZ2, self.w2.T)
        dZ1 = dA1 * (self.A1 * (1 - self.A1))
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)    # batch维度求和

        self.w1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.w2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

if __name__ == '__main__':
    mlp = MLP(4, 10, 3)
    X = np.array([5.1, 3.5, 1.4, 0.2])
    y = np.array([0])
    mlp.forward(X)
    mlp.backprop(X, y)