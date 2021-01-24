import numpy as np


def load_data_sets(X_train_path, y_train_path, X_test_path, y_test_path):
    """Loads numpy arrays for training and test sets from npy files."""
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)

    return X_train, y_train, X_test, y_test
