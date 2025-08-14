def accuracy_score(y_pred, y_real):
    correct_predictions = (y_pred == y_real).sum()
    total_predictions = len(y_real)
    accuracy = correct_predictions / total_predictions
    return accuracy