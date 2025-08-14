import numpy as np

def train_one_epoch(model, X, y, learning_rate=0.01):
    # Calculate loss
    loss = model.calculate_loss(X, y)

    # Backward pass
    model.backprop(X, y, learning_rate)

    return loss

